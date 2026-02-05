import type { CourseExercise } from "../types";
import { phase1Exercises } from "./phase-1";
import { phase2Exercises } from "./phase-2";
import { phase3Exercises } from "./phase-3";

// Re-export individual phase arrays
export { phase1Exercises } from "./phase-1";
export { phase2Exercises } from "./phase-2";
export { phase3Exercises } from "./phase-3";

// Combined array of all course exercises
export const allCourseExercises: CourseExercise[] = [
  ...phase1Exercises,
  ...phase2Exercises,
  ...phase3Exercises,
];

// Get exercises by phase
export function getExercisesByPhase(phase: 1 | 2 | 3): CourseExercise[] {
  switch (phase) {
    case 1:
      return phase1Exercises;
    case 2:
      return phase2Exercises;
    case 3:
      return phase3Exercises;
  }
}

// Get a specific exercise by id
export function getCourseExerciseById(id: string): CourseExercise | undefined {
  return allCourseExercises.find((e) => e.id === id);
}

// Get exercise by phase and order
export function getCourseExerciseByPhaseAndOrder(
  phase: 1 | 2 | 3,
  order: number
): CourseExercise | undefined {
  return getExercisesByPhase(phase).find((e) => e.order === order);
}

// Get next exercise in phase (or undefined if last)
export function getNextExercise(
  currentId: string
): CourseExercise | undefined {
  const current = getCourseExerciseById(currentId);
  if (!current) return undefined;

  const phaseExercises = getExercisesByPhase(current.phase);
  const currentIndex = phaseExercises.findIndex((e) => e.id === currentId);

  if (currentIndex < phaseExercises.length - 1) {
    return phaseExercises[currentIndex + 1];
  }

  // If at end of phase, try next phase
  if (current.phase < 3) {
    const nextPhase = (current.phase + 1) as 1 | 2 | 3;
    const nextPhaseExercises = getExercisesByPhase(nextPhase);
    return nextPhaseExercises[0];
  }

  return undefined;
}

// Get previous exercise in phase (or undefined if first)
export function getPreviousExercise(
  currentId: string
): CourseExercise | undefined {
  const current = getCourseExerciseById(currentId);
  if (!current) return undefined;

  const phaseExercises = getExercisesByPhase(current.phase);
  const currentIndex = phaseExercises.findIndex((e) => e.id === currentId);

  if (currentIndex > 0) {
    return phaseExercises[currentIndex - 1];
  }

  // If at start of phase, try previous phase
  if (current.phase > 1) {
    const prevPhase = (current.phase - 1) as 1 | 2 | 3;
    const prevPhaseExercises = getExercisesByPhase(prevPhase);
    return prevPhaseExercises[prevPhaseExercises.length - 1];
  }

  return undefined;
}

// Get total exercise count
export function getTotalExerciseCount(): number {
  return allCourseExercises.length;
}

// Get phase exercise count
export function getPhaseExerciseCount(phase: 1 | 2 | 3): number {
  return getExercisesByPhase(phase).length;
}

// Generate static params for Next.js
export function generateCourseStaticParams(): { phase: string; id: string }[] {
  return allCourseExercises.map((e) => ({
    phase: e.phase.toString(),
    id: e.id,
  }));
}
