cd ~/benchmark

BRANCHNAME=$(date +results%y%m%d)
git checkout main
git pull
git checkout -b $BRANCHNAME

./update_sources.sh

asv run --machine aws-c5.large

git add --force .asv/results
git commit -m "automatic submission of aws results"
git push --set-upstream origin $BRANCHNAME

asv gh-pages 