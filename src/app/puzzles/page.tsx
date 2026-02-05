"use client";

import Link from "next/link";
import { Puzzle, CheckCircle2, Circle } from "lucide-react";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { puzzles } from "@/data/puzzles";
import { useProgress } from "@/contexts/progress-context";

const difficultyColors = {
  easy: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  medium:
    "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  hard: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

export default function PuzzlesPage() {
  const { progress, isLoaded } = useProgress();

  const completedCount = isLoaded
    ? puzzles.filter((p) => progress.puzzles[p.id]?.completed).length
    : 0;

  return (
    <div className="container py-8 max-w-4xl">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Puzzles</h1>
            <p className="text-muted-foreground">
              Practice challenges to sharpen your scikit-learn skills.
            </p>
          </div>
          <Badge variant="outline" className="text-lg px-4 py-2">
            {completedCount}/{puzzles.length}
          </Badge>
        </div>
      </div>

      <div className="grid gap-3">
        {puzzles.map((puzzle) => {
          const isComplete = progress.puzzles[puzzle.id]?.completed;

          return (
            <Link key={puzzle.id} href={`/puzzle/${puzzle.id}`}>
              <Card className="hover:bg-muted/50 transition-colors cursor-pointer">
                <CardHeader className="py-4">
                  <div className="flex items-center gap-3">
                    {isComplete ? (
                      <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0" />
                    ) : (
                      <Circle className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <CardTitle className="text-base flex items-center gap-2">
                        <Puzzle className="h-4 w-4 text-muted-foreground" />
                        {puzzle.title}
                      </CardTitle>
                      <CardDescription className="line-clamp-1">
                        {puzzle.description}
                      </CardDescription>
                    </div>
                    {puzzle.difficulty && (
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full capitalize ${
                          difficultyColors[puzzle.difficulty]
                        }`}
                      >
                        {puzzle.difficulty}
                      </span>
                    )}
                  </div>
                </CardHeader>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
