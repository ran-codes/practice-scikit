export interface BaseExercise {
  id: string;
  title: string;
  description: string;
  code: string;
  difficulty: "easy" | "medium" | "hard";
  hints?: string[];
  expectedOutput?: string;
}

export interface CourseExercise extends BaseExercise {
  type: "course";
  phase: 1 | 2 | 3;
  order: number;
  concepts: string[];
  cheatSheet?: string;
}

export interface Puzzle extends BaseExercise {
  type: "puzzle";
}

export type Exercise = CourseExercise | Puzzle;

export interface ExerciseProgress {
  completed: boolean;
  completedAt?: string;
  code?: string;
}

export interface Progress {
  course: Record<string, ExerciseProgress>;
  puzzles: Record<string, ExerciseProgress>;
}

export const PHASE_INFO = {
  1: {
    title: "Basics",
    subtitle: "Core ML Workflow",
    description: "Learn the fundamental scikit-learn workflow: loading datasets, train/test split, fit/predict pattern, and basic evaluation.",
    totalExercises: 15,
  },
  2: {
    title: "Intermediate",
    subtitle: "Preprocessing & Pipelines",
    description: "Master cross-validation, preprocessing, pipelines, advanced metrics, and hyperparameter tuning.",
    totalExercises: 12,
  },
  3: {
    title: "Advanced",
    subtitle: "Advanced Techniques",
    description: "Explore SVMs, ensemble methods, dimensionality reduction, clustering, and complete ML pipelines.",
    totalExercises: 10,
  },
} as const;
