# Set & move to home directory
source set_env.sh
#Make sure the snorkel submodule is installed
git submodule update --init --recursive

#Change directory into snorkel
cd "$SNORKELHOME"

# Make sure the submodules are installed
git submodule update --init --recursive

# Make sure parser is installed
PARSER="$SNORKELHOME/parser/stanford-corenlp-3.6.0.jar"
if [ ! -f "$PARSER" ]; then
    read -p "CoreNLP [default] parser not found- install now?   " yn
    case $yn in
        [Yy]* ) echo "Installing parser..."; ./install-parser.sh;;
        [Nn]* ) ;;
    esac
fi

# Launch jupyter notebook!
echo "Launching Jupyter Notebook..."
cd "$WORKINGPATH"
jupyter notebook
