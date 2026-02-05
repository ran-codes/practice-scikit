# Practice Scikit-Learn

Interactive scikit-learn learning platform running entirely in the browser using Pyodide (WebAssembly Python).

**[Live Demo](https://ranfysvern.github.io/practice-scikit)**

## Features

- **51 Interactive Exercises** - Hands-on coding challenges across 3 phases
- **Browser-Based Python** - No installation required; runs via Pyodide
- **Progress Tracking** - Local storage saves your completion status
- **Documentation Links** - Direct links to official sklearn docs for each exercise
- **Solutions Included** - View solutions when you're stuck

## Course Structure

| Phase | Focus | Exercises |
|-------|-------|-----------|
| **1** | ML Fundamentals | 15 exercises covering data loading, train/test split, basic models |
| **2** | Evaluation & Preprocessing | 12 exercises on metrics, pipelines, cross-validation |
| **3** | Advanced Techniques | 10 exercises on SVM, ensemble methods, clustering, PCA |

Plus **14 standalone puzzles** for targeted practice.

## Quick Start

```bash
npm install
npm run dev     # localhost:3000
```

## Tech Stack

- **Next.js 15** - Static export to GitHub Pages
- **Pyodide 0.27** - WebAssembly Python with sklearn, pandas, numpy, matplotlib
- **shadcn/ui** - Component library
- **CodeMirror 6** - Code editor

## License

MIT
