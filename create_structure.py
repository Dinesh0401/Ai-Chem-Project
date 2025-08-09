import os

def create_from_tree(tree_lines, root_dir='.'):
    for line in tree_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        name = stripped.split(' ', 1)[-1].split('#')[0].strip()
        path = os.path.join(root_dir, name.rstrip('/'))
        if name.endswith('/'):
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').close()

tree = [
    "backend/",
    "backend/app.py",
    "backend/requirements.txt",
    "backend/models/",
    "backend/models/gvae_trained.pth",
    "backend/models/ppo_molgen.zip",
    "backend/data/",
    "backend/data/processed.pt",
    "backend/services/",
    "backend/services/model_service.py",
    "backend/services/props.py",
    "backend/services/rl_env.py",
    "backend/static/",
    "frontend/",
    "frontend/package.json",
    "frontend/tsconfig.json",
    "frontend/public/",
    "frontend/public/index.html",
    "frontend/src/",
    "frontend/src/index.tsx",
    "frontend/src/App.tsx",
    "frontend/src/styles/",
    "frontend/src/styles/app.css",
    "frontend/src/utils/",
    "frontend/src/utils/api.ts",
    "frontend/src/components/",
    "frontend/src/components/PropertySliders.tsx",
    "frontend/src/components/MoleculeCard.tsx",
    "frontend/src/components/LatentExplorer.tsx",
    "frontend/src/pages/",
    "frontend/src/pages/Home.tsx",
    "frontend/src/pages/Generate.tsx",
    "frontend/src/pages/Evaluate.tsx",
    "scripts/",
    "scripts/data_preparation.py",
    "scripts/train.py",
    ".gitignore",
    "README.md"
]

if __name__ == "__main__":
    create_from_tree(tree, root_dir="my_project")
    print("Project structure created in 'my_project'")
