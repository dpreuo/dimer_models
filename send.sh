rsync -avz --delete \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='.gitattributes' \
    /Users/perudornellas/python/packages/dimer_models/ \
    hyperion:/scratch/perudornellas/dimer_models/