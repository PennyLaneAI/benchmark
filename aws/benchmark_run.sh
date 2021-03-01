cd ~/benchmark

BRANCHNAME=$(date +results%y%m%d)
git checkout master
git checkout -b $BRANCHNAME

asv run NEW --interleave-processes --parallel --config full_dependencies.conf.json --machine aws-c5.large

git add --force .asv/results
git commit -m "automatic submission of aws results"
git push --set-upstream origin $BRANCHNAME

asv gh-pages --rewrite