rsync -avz --progress \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='.gitattributes' \
    hyperion:/scratch/perudornellas/dimer_models/ \
    /Users/perudornellas/python/packages/dimer_models/