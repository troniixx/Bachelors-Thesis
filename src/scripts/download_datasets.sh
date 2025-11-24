#!/usr/bin/env bash
# download_data.sh
# Copyright (c) 2025 Mert Erol
# Use this only if you do not have the datasets in the data/ folder yet.

set -euo pipefail

echo "⬇️  Downloading datasets from Google Drive..."

# Ensure 'gdown' is available. Try to install it to the user's Python environment if missing.
ensure_gdown() {
	if command -v gdown >/dev/null 2>&1; then
		return 0
	fi

	echo "gdown: command not found — attempting to install via pip (user scope)..."
	if command -v python3 >/dev/null 2>&1; then
		# Try user install first
		if python3 -m pip install --user gdown; then
			# If installed to user base, make sure user-base bin is on PATH for this run
			if python3 -m site >/dev/null 2>&1; then
				USER_BASE_BIN="$(python3 -m site --user-base 2>/dev/null)/bin"
				if [ -d "$USER_BASE_BIN" ]; then
					export PATH="$USER_BASE_BIN:$PATH"
				fi
			fi
		else
			echo "pip install --user gdown failed; trying a global install..."
			if python3 -m pip install gdown; then
				echo "gdown installed globally."
			else
				echo "Automatic installation failed."
			fi
		fi
	else
		echo "python3 not found on PATH; cannot install gdown automatically."
	fi

	if ! command -v gdown >/dev/null 2>&1; then
		echo
		echo "ERROR: 'gdown' is still not available."
		echo "Options to proceed:" 
		echo "  1) Install gdown locally: python3 -m pip install --user gdown"
		echo "     (You may need to add \"$(python3 -m site --user-base 2>/dev/null)/bin\" to your PATH.)"
		echo "  2) Install with pipx if you use pipx: pipx install gdown"
		echo "  3) Install system-wide: python3 -m pip install gdown"
		echo
		echo "After installing, re-run this script."
		exit 1
	fi
}

# Run the ensure step, then call gdown to download the folder
ensure_gdown

# Target directory (will be created if missing). The goal: download files directly into this
# existing data folder (do NOT create an extra nested folder).
TARGET_DIR="data"

if [ -e "$TARGET_DIR" ] && [ ! -d "$TARGET_DIR" ]; then
	echo "ERROR: '$TARGET_DIR' exists and is not a directory. Move or remove that file and re-run."
	exit 1
fi

mkdir -p "$TARGET_DIR"

# Use a temporary subdir to let gdown create its folder(s) there, then move the contents up
# so files end up directly in $TARGET_DIR (this avoids nested folder names).
TMPDIR="$TARGET_DIR/.gdown_tmp_$$"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

echo "Downloading into temporary directory: $TMPDIR"
# Replace the link below with your folder URL
gdown --folder "https://drive.google.com/drive/folders/1dAnc6ukPf0kC3rjqlSAoTipZnS7uSxHH?usp=sharing" -O "$TMPDIR"

echo "Moving downloaded files into $TARGET_DIR (existing files may be overwritten)..."
# Move all files (including dotfiles) from TMPDIR into TARGET_DIR
(
	# enable dotglob so hidden files are moved as well; nullglob avoids literal globs if empty
	shopt -s dotglob nullglob
	for f in "$TMPDIR"/*; do
		mv -f "$f" "$TARGET_DIR"/
	done
)

# Clean up temp dir
rm -rf "$TMPDIR"

echo "All datasets downloaded to $TARGET_DIR/"