# Practice Scikit-Learn

Interactive scikit-learn learning platform running entirely in the browser using Pyodide.

## Quick Start

```bash
npm install
npm run dev     # Development server at localhost:3000
npm run build   # Static export to ./out
```

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
├── components/             # React components
├── contexts/               # Pyodide and progress contexts
├── data/                   # Course exercises and puzzles
│   ├── course/             # Phase 1-3 exercises
│   └── puzzles.ts          # Standalone puzzles
└── lib/                    # Utilities (run-python.ts)
```

## Key Commands

- `npm run dev` - Start development server
- `npm run build` - Build static site
- `npm run lint` - Run ESLint

---

<!-- MAINTAINER DETAILS BELOW -->

## Architecture Overview

### Runtime Stack
- **Pyodide 0.27.0**: WebAssembly Python runtime
- **Packages loaded**: scikit-learn, pandas, numpy, matplotlib
- **Execution**: All Python runs client-side via `src/lib/run-python.ts`

### Data Flow
1. User writes code in CodeMirror editor
2. Code sent to Pyodide via `usePyodide()` hook
3. Output captured (stdout + matplotlib figures + result variable)
4. HTML rendered in OutputPanel component

### Progress System
- Uses localForage (IndexedDB wrapper)
- Storage key: `sklearn-learning-progress`
- Tracks completion per exercise/puzzle with timestamps

## Adding Content

### Adding a New Exercise

Edit the appropriate phase file in `src/data/course/`:

```typescript
// src/data/course/phase-1.ts
{
  id: "unique-slug",           // URL-safe, unique across all phases
  type: "course",
  phase: 1,                    // 1, 2, or 3
  order: 16,                   // Sequential within phase
  title: "Exercise Title",
  description: "What the user should accomplish",
  difficulty: "easy",          // easy | medium | hard
  concepts: ["concept1", "concept2"],
  hints: ["Hint 1", "Hint 2"],
  code: `# Starter code here
# TODO comments guide the user
result = None  # <- modify this line
result`,
}
```

**Important**: Update `PHASE_INFO.totalExercises` in `src/data/types.ts` when adding exercises.

### Adding a New Puzzle

Edit `src/data/puzzles.ts`:

```typescript
{
  id: "unique-puzzle-slug",
  title: "Puzzle Title",
  description: "Task description",
  difficulty: "medium",
  context: "Why this matters and real-world applications",
  code: `# Starter code
# TODO: Complete the task
result = None
result`,
}
```

### Code Template Pattern

All exercises/puzzles should:
1. Import necessary modules at the top
2. Set up data/context
3. Have clear `# TODO` comments marking what to complete
4. Assign final answer to `result` variable
5. End with `result` to display output

## Component Reference

| Component | Purpose |
|-----------|---------|
| `PyodideProvider` | Loads Pyodide, provides `runPython()` |
| `ProgressProvider` | Manages completion state |
| `ExercisePage` | Course exercise UI with hints, navigation |
| `PuzzlePage` | Standalone puzzle UI |
| `CodeEditor` | CodeMirror 6 wrapper |
| `OutputPanel` | Renders Python output + matplotlib |

## Matplotlib Support

Plots are automatically captured. The `run-python.ts` module:
1. Captures all matplotlib figures after code execution
2. Converts each to base64 PNG
3. Injects as `<img>` tags in output HTML
4. Closes figures to free memory

Users don't need special code - just `plt.show()` or let figures exist.

## Deployment

GitHub Actions workflow at `.github/workflows/nextjs.yml`:
- Triggers on push to `main`
- Builds static site with `npm run build`
- Deploys `./out` directory to GitHub Pages

### Manual Deployment
```bash
npm run build
# Upload ./out to any static host
```

## Troubleshooting

### Pyodide Loading Issues
- Check browser console for network errors
- Pyodide CDN: `cdn.jsdelivr.net/pyodide/v0.27.0/full/`
- Large packages (sklearn ~15MB) may take time on slow connections

### Build Errors
- Ensure all exercise IDs are unique
- Check `generateStaticParams()` returns valid paths
- Verify TypeScript types match exercise structure

### Progress Not Saving
- Check browser IndexedDB isn't disabled
- localForage falls back to localStorage if needed
- Clear with: `localforage.removeItem('sklearn-learning-progress')`

## Content Guidelines

### Difficulty Levels
- **Easy**: Single concept, minimal code changes, clear instructions
- **Medium**: Multiple concepts combined, some problem-solving required
- **Hard**: Complex workflows, minimal guidance, real-world patterns

### Exercise Progression
- Phase 1: Learn the sklearn API (fit/predict pattern)
- Phase 2: Master evaluation and preprocessing
- Phase 3: Advanced techniques and complete workflows

### Writing Good Hints
1. First hint: Point to the right function/method
2. Second hint: Explain the key parameter
3. Third hint: Show the general pattern (without giving away the answer)
