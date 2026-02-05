"use client";

import Link from "next/link";
import { BookOpen, Puzzle, Keyboard, ArrowRight, GraduationCap, Rocket } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { puzzles } from "@/data/puzzles";
import { PHASE_INFO } from "@/data/types";
import { getTotalExerciseCount, getExercisesByPhase } from "@/data/course";
import { useProgress } from "@/contexts/progress-context";

const PHASE_ICONS = {
  1: BookOpen,
  2: GraduationCap,
  3: Rocket,
};

export default function Home() {
  const { progress, isLoaded } = useProgress();

  const totalCourseExercises = getTotalExerciseCount();
  const completedCourseCount = isLoaded
    ? Object.values(progress.course).filter((p) => p.completed).length
    : 0;
  const completedPuzzleCount = isLoaded
    ? Object.values(progress.puzzles).filter((p) => p.completed).length
    : 0;

  const coursePercentage = Math.round((completedCourseCount / totalCourseExercises) * 100);
  const puzzlePercentage = puzzles.length > 0
    ? Math.round((completedPuzzleCount / puzzles.length) * 100)
    : 0;

  return (
    <div className="container py-8 max-w-5xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Scikit-Learn Practice</h1>
        <p className="text-muted-foreground">
          Master machine learning with scikit-learn through structured lessons and practice challenges.
        </p>
      </div>

      {/* Main Cards - Course & Puzzles */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Course Card */}
        <Card className="relative overflow-hidden">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <BookOpen className="h-5 w-5 text-primary" />
              </div>
              <div>
                <CardTitle>Course</CardTitle>
                <CardDescription>Structured learning path</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {totalCourseExercises} exercises across 3 phases, from basics to advanced ML techniques.
              </p>
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Progress</span>
                  <span>{completedCourseCount}/{totalCourseExercises} ({coursePercentage}%)</span>
                </div>
                <Progress value={coursePercentage} className="h-2" />
              </div>
              <Link href="/course">
                <Button className="w-full">
                  Continue Learning
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Puzzles Card */}
        <Card className="relative overflow-hidden">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-orange-500/10">
                <Puzzle className="h-5 w-5 text-orange-500" />
              </div>
              <div>
                <CardTitle>Puzzles</CardTitle>
                <CardDescription>Practice challenges</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {puzzles.length} standalone puzzles to test your ML skills. New puzzles added regularly!
              </p>
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Completed</span>
                  <span>{completedPuzzleCount}/{puzzles.length} ({puzzlePercentage}%)</span>
                </div>
                <Progress value={puzzlePercentage} className="h-2" />
              </div>
              <Link href="/puzzles">
                <Button variant="outline" className="w-full">
                  View Puzzles
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Course Phase Overview */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Course Phases</h2>
        <div className="grid sm:grid-cols-3 gap-4">
          {([1, 2, 3] as const).map((phase) => {
            const info = PHASE_INFO[phase];
            const Icon = PHASE_ICONS[phase];
            const exercises = getExercisesByPhase(phase);
            const completed = isLoaded
              ? exercises.filter((e) => progress.course[e.id]?.completed).length
              : 0;

            return (
              <Link key={phase} href={`/course/${phase}`}>
                <Card className="hover:bg-muted/50 transition-colors cursor-pointer h-full">
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-2">
                      <Icon className="h-4 w-4 text-muted-foreground" />
                      <CardDescription>Phase {phase}</CardDescription>
                    </div>
                    <CardTitle className="text-lg">{info.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-xs text-muted-foreground mb-2">
                      {completed}/{info.totalExercises} completed
                    </p>
                    <Progress
                      value={Math.round((completed / info.totalExercises) * 100)}
                      className="h-1.5"
                    />
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Keyboard Shortcuts */}
      <Card className="bg-muted/50">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Keyboard className="h-5 w-5" />
            <CardTitle className="text-base">Keyboard Shortcuts</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Command palette</span>
              <kbd className="px-2 py-1 bg-background rounded border text-xs font-mono">
                Ctrl+K
              </kbd>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Run code</span>
              <kbd className="px-2 py-1 bg-background rounded border text-xs font-mono">
                Ctrl+Enter
              </kbd>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Reset code</span>
              <kbd className="px-2 py-1 bg-background rounded border text-xs font-mono">
                Ctrl+Shift+R
              </kbd>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Toggle sidebar</span>
              <kbd className="px-2 py-1 bg-background rounded border text-xs font-mono">
                Ctrl+B
              </kbd>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
