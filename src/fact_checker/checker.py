import re, json, math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse, unquote
import idna
import tldextract

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.

# ---------- Utilities ----------

# NOTE: This regex is intentionally simple and fast. It will catch most HTTP(S) URLs
# but is not a fully RFC-3986 compliant parser (e.g., it may miss edge cases or
# accept some malformed inputs). Prefer using urllib.parse for deeper parsing.
URL_REGEX = re.compile(
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b"
    r"(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

def extract_urls(text: str) -> List[str]:
    """Extract de-duplicated HTTP(S) URLs from a block of text.

    This function uses a pragmatic URL regex to find links, then:
        - Ensures each URL has a scheme (http/https). (Regex already enforces it.)
        - Strips common trailing punctuation like ``) . , ; ! ? " '`` that often
        stick to URLs in prose.
        - Preserves first occurrence order while removing duplicates.

    Args:
        text: Arbitrary text to scan.

    Returns:
        A list of unique URLs in the order they first appear.

    Examples:
        >>> extract_urls("see https://example.com, also http://a.co/path.")
        ['https://example.com', 'http://a.co/path']
    """
    urls = URL_REGEX.findall(text or "")

    out: List[str] = []
    for u in urls:
        # Regex already matches http/https, but keep this guard in case you later
        # change URL_REGEX or feed pre-parsed strings.
        if not u.lower().startswith(("http://", "https://")):
            u = "http://" + u

        # Trim stray punctuation that often trails URLs in natural text.
        u = u.strip(").,;!?\"'")
        out.append(u)

    # De-duplicate while preserving order.
    return list(dict.fromkeys(out))


def tld_extract(domain: str) -> str:
    """Return the registered domain (eTLD+1) component of a host.

    Uses ``tldextract`` to robustly split a host into subdomain, domain, and suffix.
    If the input is not recognized as a registered domain (e.g., localhost, IP),
    the original lowercased input is returned.

    Args:
        domain: A hostname (may include subdomains).

    Returns:
        The registered domain (e.g., ``example.co.uk``), or the lowercased input
        if no registered domain is found.

    Examples:
        >>> tld_extract("a.b.example.co.uk")
        'example.co.uk'
        >>> tld_extract("localhost")
        'localhost'
    """
    ext = tldextract.extract(domain)
    if not ext.top_domain_under_public_suffix:
        return domain.lower()
    return ext.top_domain_under_public_suffix.lower()


def domain_from_url(url: str) -> Optional[str]:
    """Extract and normalize the hostname from a URL.

    - Uses ``urllib.parse.urlparse`` to obtain the hostname.
    - Decodes Punycode (``xn--``) into Unicode using ``idna.decode``.
    - Lowercases the result.

    Args:
        url: A URL string (e.g., ``https://sub.xn--d1acpjx3f.xn--p1ai/path``).

    Returns:
        The decoded, lowercased hostname, or ``None`` if parsing fails or the
        URL has no hostname.

    Examples:
        >>> domain_from_url("https://www.Example.com/path")
        'www.example.com'
    """
    try:
        host = urlparse(url).hostname
        if not host:
            return None
        # Decode Punycode if present.
        decoded = idna.decode(host) if host.startswith("xn--") else host
        return decoded.lower()
    except Exception:
        return None


def count_subdomains(domain: str) -> int:
    """Count the number of subdomain labels in a hostname.

    For ``a.b.example.co.uk`` the subdomain part is ``a.b`` → 2 labels.

    Args:
        domain: A hostname string.

    Returns:
        Number of subdomain labels (0 if none).

    Examples:
        >>> count_subdomains("a.b.example.com")
        2
        >>> count_subdomains("example.com")
        0
    """
    ext = tldextract.extract(domain)
    if ext.subdomain:
        return len(ext.subdomain.split("."))
    return 0


def is_ip_literal(domain: str) -> bool:
    """Heuristically check whether a hostname is an IPv4 literal.

    This is a simple pattern check and does **not** validate octet ranges (0–255).

    Args:
        domain: Host portion of a URL or a bare string.

    Returns:
        True if the string looks like ``A.B.C.D`` where each part is 1–3 digits,
        otherwise False.

    Examples:
        >>> is_ip_literal("192.168.0.1")
        True
        >>> is_ip_literal("example.com")
        False
    """
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))


def has_hex_encoding(url: str) -> bool:
    """Detect URL-encoded (percent-encoded) bytes within a URL string.

    Checks for common encodings such as ``%3A`` (``:``) and ``%2F`` (``/``),
    as well as any ``%HH`` hex pattern.

    Args:
        url: The URL to inspect.

    Returns:
        True if any percent-encoded bytes are present, else False.

    Examples:
        >>> has_hex_encoding("https://ex.com/a%2Fb")
        True
        >>> has_hex_encoding("https://ex.com/path")
        False
    """
    lower = url.lower()
    return "%3a" in lower or "%2f" in lower or re.search(r"%[0-9a-fA-F]{2}", url) is not None


def has_at_symbol(url: str) -> bool:
    """Check whether the URL's network location contains an '@' sign.

    URLs of the form ``scheme://user@host`` can be misleading to users
    (everything before '@' may be interpreted as credentials or noise).

    Args:
        url: The URL to inspect.

    Returns:
        True if ``@`` appears in the netloc (authority) component, else False.

    Examples:
        >>> has_at_symbol("http://user@evil.com")
        True
        >>> has_at_symbol("http://example.com")
        False
    """
    return "@" in urlparse(url).netloc

# ---------- Knowledge Base Loaders ----------

def load_brands(path="data/fact_checking/brands.csv") -> Dict[str, str]:
    """Load brands from CSV file

    Args:
        path (str): Path to CSV file

    Returns:
        Dict[str, str]: Brands
    """
    base = {}
    
    try:
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                base[row["brand"].strip().lower()] = row["official_domain"].strip().lower()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return base

def load_tld_risks(path="data/fact_checking/clean_tlds.csv") -> Dict[str, str]:
    """Load TLD risks from CSV file

    Args:
        path (str): Path to CSV file

    Returns:
        Dict[str, str]: TLD risks
    """
    risks = {}
    
    try:
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                risks[row["metadata_tld"].strip().lower()] = row["metadata_severity"].strip().lower()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return risks

def load_phrases(path="data/fact_checking/phrases.json") -> Dict[str, List[str]]:
    """Load phrases from JSON fileß

    Args:
        path (str): Path to JSON file

    Returns:
        Dict[str, List[str]: Phrases
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
SEVERITY_SCORES = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
    "critical": 1.0
}

@dataclass
class Evidence:
    label: str
    details: Dict[str, Any]
    
@dataclass
class FactCheckResult:
    fact_risk: float
    components: Dict[str, float]
    evidence: List[Evidence]
    message: List[str]
    
def etld1(domain: str) -> str:
    """Extract eTLD+1 from a domain, or return the original domain if not possible.

    Args:
        domain (str): The domain to extract from.

    Returns:
        str: The eTLD+1 or the original domain.
    """
    ext = tldextract.extract(domain)
    if not ext.registered_domain:
        return domain.lower()
    return ext.registered_domain.lower()

# ---------- Checker ----------

class FactChecker:
    def __init__(self):
        self.brands = load_brands()
        self.tld_risks = load_tld_risks()
        self.phrases = load_phrases()
        
    def extract_entities_and_claims(self, text:str) -> Tuple[List[str], Dict[str, List[str]]]:
        # TODO: Add NER with spacy or similar
        # NOTE: For now: Minimal approach with regex and phrase matching
        brands_found = []
        lowered = text.lower()
        
        for brand in self.brands.keys():
            if brand in lowered:
                brands_found.append(brand)
        
        claims = {k: [] for k in self.phrases}
        for cat, plist in self.phrases.items():
            for phrase in plist:
                if phrase.lower() in lowered:
                    claims[cat].append(phrase)
        
        return list(dict.fromkeys(brands_found)), claims

    def check_urls(self, urls: List[str]) -> Tuple[float, List[Evidence], Dict[str, float]]:
        obf_score = 0.0
        tld_score = 0.0
        evidence = []
        
        for u in urls:
            d = domain_from_url(u)
            if not d:
                continue
            
            # obfuscation checks
            flags = {
                "punycode": d.startswith("xn--"),
                "ip_literal": is_ip_literal(d),
                "hex_encoding": has_hex_encoding(u),
                "at_symbol": has_at_symbol(u),
                "subdomain_count": count_subdomains(d) > 2
            }
            
            if any(flags.values()):
                obf_score = max(obf_score, 0)
                evidence.append(Evidence("urkl_obfuscation", {"url": u, **flags}))
            
            # tld risk check
            ext = tldextract.extract(d)
            tld = (ext.suffix or "").lower()
            
            if tld:
                severity = self.tld_risks.get(tld)
                if severity:
                    score = SEVERITY_SCORES.get(severity, 0.0)
                    tld_score = max(tld_score, score)
                    evidence.append(Evidence("tld_risk", {"url": u, "tld": tld, "severity": severity, "score": score}))
                    
        return max(obf_score, 0.0), evidence, {"url_obfuscation": obf_score, "tld_risk": tld_score}
    
    def check_brand_impersonation(self, brands_found: List[str], urls: List[str], sender_email: Optional[str] = None) -> Tuple[float, List[Evidence], Optional[str]]:
        sender_domain_etld1 = None
        if sender_email and "@" in sender_email:
            sender_domain_etld1 = etld1(sender_email.split("@")[-1])
            
        link_domains = [etld1(domain_from_url(u) or "") for u in urls if domain_from_url(u)]
        mismatch = 0.0
        evidence = []
        
        for brand in brands_found:
            officials = self.brands.get(brand)
            if not officials:
                continue
            
            official_etld1 = etld1(officials)
            bad_links = [d for d in link_domains if d and d != official_etld1]
            sender_mismatch = (sender_domain_etld1 and sender_domain_etld1 != official_etld1)
            
            if bad_links or sender_mismatch:
                mismatch = 1.0
                evidence.append(Evidence("brand_impersonation", {
                    "brand": brand,
                    "official_domain": official_etld1,
                    "sender_domain": sender_domain_etld1,
                    "link_domains": list(sorted(set(bad_links))),
                }))

        return mismatch, evidence, sender_domain_etld1
    
    def claim_risk_score(self, claim_hits: Dict[str, List[str]]) -> Tuple[float, List[Evidence]]:
        pass
    
    def check(self, email_text: str, sender_email: Optional[str] = None) -> FactCheckResult:
        pass
        
# ---------- Runner ----------

if __name__ == "__main__":
    # Testing utilities (general functionality, removing duplicates and preserving order)
    text = "Check out https://example.com/path?query=1 and http://sub.domain.co.uk! Also visit https://example.com/path?query=1."
    urls = extract_urls(text)
    print("Extracted URLs:", urls)
    for url in urls:
        domain = domain_from_url(url)
        tld = tld_extract(domain)
        subdomain_count = count_subdomains(domain)
        ip_check = is_ip_literal(domain)
        hex_check = has_hex_encoding(url)
        at_check = has_at_symbol(url)
        
        print(f"URL: {url}")
        print(f"  Domain: {domain}")
        print(f"  TLD: {tld}")
        print(f"  Subdomain Count: {subdomain_count}")
        print(f"  Is IP Literal: {ip_check}")
        print(f"  Has Hex Encoding: {hex_check}")
        print(f"  Has '@' Symbol: {at_check}")
        

    # Testing loaders
    tlds = load_tld_risks()
    brands = load_brands()
    phrases = load_phrases()
    
    print(f"Loaded {len(tlds)} TLD risks")
    print(f"Loaded {len(brands)} brands")
    print(f"Loaded {len(phrases)} phrase categories")
    
    print("Work in progress")
    
    # Main checker test
    email_text = """
    Dear user, your PayPal account is locked. Verify your account within 24 hours:
    https://xn--paypa1-secure-login.example.tk/login
    """
    fc = FactChecker()
