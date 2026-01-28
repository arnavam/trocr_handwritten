#!/bin/bash
# Install Kraken in a separate conda environment
# Kraken requires Python ≤3.11 and torch <2.5

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Creating kraken_env conda environment with Python 3.11...${NC}"
conda create -n kraken_env python=3.11 -y

echo -e "${GREEN}Activating kraken_env...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kraken_env

echo -e "${GREEN}Installing kraken...${NC}"
pip install kraken

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}To use kraken:${NC}"
echo -e "  conda activate kraken_env"
echo -e "  kraken -i image.jpg segment -bl -o output.json"
echo -e ""
echo -e "${GREEN}Or use the wrapper script:${NC}"
echo -e "  python -m trocr_handwritten.segmentation.kraken_segment -i image.jpg -o output/"
