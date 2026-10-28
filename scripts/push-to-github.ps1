git init
git add .
git commit -m "Initial Tradex RL trading platform"
git branch -M main
gh repo create Tradex --public --source=. --remote=origin --push
