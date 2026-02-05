"use client";

import Link from "next/link";
import { ChevronLeft, CheckCircle2, Circle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useProgress } from "@/contexts/progress-context";
import type { CourseExercise } from "@/data/types";

interface PhaseInfo {
  title: string;
  subtitle: string;
  description: string;
  totalExercises: number;
}

interface Props {
  phase: 1 | 2 | 3;
  info: PhaseInfo;
  exercises: CourseExercise[];
}

export function PhasePageClient({ phase, info, exercises }: Props) {
  const { progress, isLoaded } = useProgress();

  const completedCount = isLoaded
    ? exercises.filter((e) => progress.course[e.id]?.completed).length
    : 0;

  return (
    <div className="container py-8 max-w-4xl">
      <div className="mb-6">
        <Link href="/course">
          <Button variant="ghost" size="sm" className="mb-4">
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Course
          </Button>
        </Link>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-1">
              Phase {phase}: {info.title}
            </h1>
            <p className="text-muted-foreground">{info.subtitle}</p>
          </div>
          <Badge variant="outline" className="text-lg px-4 py-2">
            {completedCount}/{exercises.length}
          </Badge>
        </div>
      </div>

      <p className="text-muted-foreground mb-6">{info.description}</p>

      <div className="grid gap-3">
        {exercises.map((exercise) => {
          const isComplete = progress.course[exercise.id]?.completed;

          return (
            <Link key={exercise.id} href={`/course/${phase}/${exercise.id}`}>
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
                        <span className="text-muted-foreground font-normal">
                          {exercise.order}.
                        </span>
                        {exercise.title}
                      </CardTitle>
                      <CardDescription className="line-clamp-1">
                        {exercise.description}
                      </CardDescription>
                    </div>
                    <Badge
                      variant={
                        exercise.difficulty === "easy"
                          ? "secondary"
                          : exercise.difficulty === "medium"
                          ? "default"
                          : "destructive"
                      }
                    >
                      {exercise.difficulty}
                    </Badge>
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
