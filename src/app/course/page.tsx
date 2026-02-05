"use client";

import Link from "next/link";
import { BookOpen, GraduationCap, Rocket, CheckCircle2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { PHASE_INFO } from "@/data/types";
import { getExercisesByPhase } from "@/data/course";
import { useProgress } from "@/contexts/progress-context";

const PHASE_ICONS = {
  1: BookOpen,
  2: GraduationCap,
  3: Rocket,
};

export default function CoursePage() {
  const { progress, isLoaded } = useProgress();

  const getPhaseCompletedCount = (phase: 1 | 2 | 3) => {
    const exercises = getExercisesByPhase(phase);
    return exercises.filter((e) => progress.course[e.id]?.completed).length;
  };

  return (
    <div className="container py-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Scikit-Learn Course</h1>
        <p className="text-muted-foreground">
          Master machine learning with scikit-learn through structured exercises. Complete all three phases to become proficient.
        </p>
      </div>

      <div className="grid gap-6">
        {([1, 2, 3] as const).map((phase) => {
          const info = PHASE_INFO[phase];
          const Icon = PHASE_ICONS[phase];
          const completed = isLoaded ? getPhaseCompletedCount(phase) : 0;
          const total = info.totalExercises;
          const percentage = Math.round((completed / total) * 100);

          return (
            <Link key={phase} href={`/course/${phase}`}>
              <Card className="hover:bg-muted/50 transition-colors cursor-pointer">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Icon className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          Phase {phase}: {info.title}
                          {completed === total && (
                            <CheckCircle2 className="h-5 w-5 text-green-500" />
                          )}
                        </CardTitle>
                        <CardDescription>{info.subtitle}</CardDescription>
                      </div>
                    </div>
                    <Badge variant="secondary">
                      {completed}/{total}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    {info.description}
                  </p>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Progress</span>
                      <span>{percentage}%</span>
                    </div>
                    <Progress value={percentage} className="h-2" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
