export SNORKELHOME="`pwd`/snorkel"
echo "Snorkel home directory: $SNORKELHOME"
export PYTHONPATH="$PYTHONPATH:$SNORKELHOME:$SNORKELHOME/treedlib"
export PATH="$PATH:$SNORKELHOME:$SNORKELHOME/treedlib"
export WORKINGPATH="$HOME/Documents/snorkeling"
echo "$PYTHONPATH"
echo "Environment variables set!"
