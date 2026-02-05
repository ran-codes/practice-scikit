"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import localforage from "localforage";
import type { Progress, ExerciseProgress } from "@/data/types";

const STORAGE_KEY = "sklearn-learning-progress";

const DEFAULT_PROGRESS: Progress = {
  course: {},
  puzzles: {},
};

interface ProgressContextValue {
  progress: Progress;
  isLoaded: boolean;
  markComplete: (
    type: "course" | "puzzles",
    id: string,
    code?: string
  ) => void;
  isComplete: (type: "course" | "puzzles", id: string) => boolean;
  getProgress: (
    type: "course" | "puzzles",
    id: string
  ) => ExerciseProgress | undefined;
  saveCode: (type: "course" | "puzzles", id: string, code: string) => void;
  getSavedCode: (type: "course" | "puzzles", id: string) => string | undefined;
  getCompletedCount: (type: "course" | "puzzles", phaseFilter?: number) => number;
  resetProgress: () => void;
}

const ProgressContext = createContext<ProgressContextValue | null>(null);

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<Progress>(DEFAULT_PROGRESS);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load progress from localforage on mount
  useEffect(() => {
    localforage
      .getItem<Progress>(STORAGE_KEY)
      .then((stored) => {
        if (stored) {
          setProgress(stored);
        }
        setIsLoaded(true);
      })
      .catch((err) => {
        console.error("Failed to load progress:", err);
        setIsLoaded(true);
      });
  }, []);

  // Save progress to localforage whenever it changes
  useEffect(() => {
    if (isLoaded) {
      localforage.setItem(STORAGE_KEY, progress).catch((err) => {
        console.error("Failed to save progress:", err);
      });
    }
  }, [progress, isLoaded]);

  const markComplete = useCallback(
    (type: "course" | "puzzles", id: string, code?: string) => {
      setProgress((prev) => ({
        ...prev,
        [type]: {
          ...prev[type],
          [id]: {
            completed: true,
            completedAt: new Date().toISOString(),
            code: code || prev[type][id]?.code,
          },
        },
      }));
    },
    []
  );

  const isComplete = useCallback(
    (type: "course" | "puzzles", id: string): boolean => {
      return progress[type][id]?.completed ?? false;
    },
    [progress]
  );

  const getProgress = useCallback(
    (type: "course" | "puzzles", id: string): ExerciseProgress | undefined => {
      return progress[type][id];
    },
    [progress]
  );

  const saveCode = useCallback(
    (type: "course" | "puzzles", id: string, code: string) => {
      setProgress((prev) => ({
        ...prev,
        [type]: {
          ...prev[type],
          [id]: {
            ...prev[type][id],
            completed: prev[type][id]?.completed ?? false,
            code,
          },
        },
      }));
    },
    []
  );

  const getSavedCode = useCallback(
    (type: "course" | "puzzles", id: string): string | undefined => {
      return progress[type][id]?.code;
    },
    [progress]
  );

  const getCompletedCount = useCallback(
    (type: "course" | "puzzles", phaseFilter?: number): number => {
      const items = progress[type];
      if (type === "puzzles" || phaseFilter === undefined) {
        return Object.values(items).filter((p) => p.completed).length;
      }
      return Object.entries(items).filter(
        ([_, p]) => p.completed
      ).length;
    },
    [progress]
  );

  const resetProgress = useCallback(() => {
    setProgress(DEFAULT_PROGRESS);
    localforage.removeItem(STORAGE_KEY);
  }, []);

  return (
    <ProgressContext.Provider
      value={{
        progress,
        isLoaded,
        markComplete,
        isComplete,
        getProgress,
        saveCode,
        getSavedCode,
        getCompletedCount,
        resetProgress,
      }}
    >
      {children}
    </ProgressContext.Provider>
  );
}

export function useProgress() {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error("useProgress must be used within a ProgressProvider");
  }
  return context;
}
