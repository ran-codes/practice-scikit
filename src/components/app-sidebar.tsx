"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  Puzzle,
  ChevronRight,
  ChevronDown,
  BookOpen,
  GraduationCap,
  Rocket,
  CheckCircle2,
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
  SidebarRail,
} from "@/components/ui/sidebar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ThemeToggle } from "@/components/theme-toggle";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { puzzles } from "@/data/puzzles";
import { PHASE_INFO } from "@/data/types";
import { getExercisesByPhase, getTotalExerciseCount } from "@/data/course";
import { usePyodide } from "@/contexts/pyodide-context";
import { useProgress } from "@/contexts/progress-context";

const PHASE_ICONS = {
  1: BookOpen,
  2: GraduationCap,
  3: Rocket,
};

export function AppSidebar() {
  const pathname = usePathname();
  const { status } = usePyodide();
  const { progress, isLoaded } = useProgress();

  const totalCourseExercises = getTotalExerciseCount();
  const completedCourseCount = isLoaded
    ? Object.values(progress.course).filter((p) => p.completed).length
    : 0;
  const completedPuzzleCount = isLoaded
    ? Object.values(progress.puzzles).filter((p) => p.completed).length
    : 0;

  const getPhaseCompletedCount = (phase: 1 | 2 | 3) => {
    const exercises = getExercisesByPhase(phase);
    return exercises.filter((e) => progress.course[e.id]?.completed).length;
  };

  const getStatusProgress = () => {
    switch (status) {
      case "idle":
        return 0;
      case "loading-runtime":
        return 33;
      case "loading-packages":
        return 66;
      case "ready":
        return 100;
      case "error":
        return 0;
      default:
        return 0;
    }
  };

  const getStatusLabel = () => {
    switch (status) {
      case "idle":
        return "Initializing...";
      case "loading-runtime":
        return "Loading Python...";
      case "loading-packages":
        return "Loading packages...";
      case "ready":
        return "Ready";
      case "error":
        return "Error loading";
      default:
        return "";
    }
  };

  const isCourseActive = pathname.startsWith("/course");
  const isPuzzlesActive = pathname.startsWith("/puzzle");

  return (
    <Sidebar>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link href="/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <BookOpen className="size-4" />
                </div>
                <div className="flex flex-col gap-0.5 leading-none">
                  <span className="font-semibold">Scikit-Learn</span>
                  <span className="text-xs text-muted-foreground">
                    {totalCourseExercises} exercises
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        {/* Navigation */}
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={pathname === "/"}>
                  <Link href="/">
                    <Home className="size-4" />
                    <span>Home</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Course Section */}
        <SidebarGroup>
          <Collapsible defaultOpen={isCourseActive} className="group/collapsible">
            <SidebarGroupLabel asChild>
              <CollapsibleTrigger className="flex w-full items-center">
                <BookOpen className="mr-2 size-4" />
                Course
                <Badge variant="outline" className="ml-auto mr-2 text-xs">
                  {completedCourseCount}/{totalCourseExercises}
                </Badge>
                <ChevronDown className="size-4 transition-transform group-data-[state=open]/collapsible:rotate-180" />
              </CollapsibleTrigger>
            </SidebarGroupLabel>
            <CollapsibleContent>
              <SidebarGroupContent>
                <SidebarMenu className="pl-2">
                  {([1, 2, 3] as const).map((phase) => {
                    const info = PHASE_INFO[phase];
                    const Icon = PHASE_ICONS[phase];
                    const completed = getPhaseCompletedCount(phase);
                    const exercises = getExercisesByPhase(phase);
                    const isPhaseActive = pathname.startsWith(`/course/${phase}`);

                    return (
                      <Collapsible
                        key={phase}
                        defaultOpen={isPhaseActive}
                        className="group/phase"
                      >
                        <SidebarMenuItem>
                          <CollapsibleTrigger asChild>
                            <SidebarMenuButton>
                              <Icon className="size-4" />
                              <span>{info.title}</span>
                              <Badge
                                variant={completed === info.totalExercises ? "default" : "outline"}
                                className="ml-auto text-xs"
                              >
                                {completed}/{info.totalExercises}
                              </Badge>
                              <ChevronRight className="size-4 transition-transform group-data-[state=open]/phase:rotate-90" />
                            </SidebarMenuButton>
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <SidebarMenuSub>
                              {exercises.map((exercise) => {
                                const isActive = pathname === `/course/${phase}/${exercise.id}`;
                                const isComplete = progress.course[exercise.id]?.completed;

                                return (
                                  <SidebarMenuSubItem key={exercise.id}>
                                    <SidebarMenuSubButton asChild isActive={isActive}>
                                      <Link href={`/course/${phase}/${exercise.id}`}>
                                        {isComplete ? (
                                          <CheckCircle2 className="size-3 text-green-500 flex-shrink-0" />
                                        ) : (
                                          <span className="size-3 flex-shrink-0 text-muted-foreground text-xs">
                                            {exercise.order}
                                          </span>
                                        )}
                                        <span className="truncate">{exercise.title}</span>
                                      </Link>
                                    </SidebarMenuSubButton>
                                  </SidebarMenuSubItem>
                                );
                              })}
                            </SidebarMenuSub>
                          </CollapsibleContent>
                        </SidebarMenuItem>
                      </Collapsible>
                    );
                  })}
                </SidebarMenu>
              </SidebarGroupContent>
            </CollapsibleContent>
          </Collapsible>
        </SidebarGroup>

        {/* Puzzles Section */}
        <SidebarGroup>
          <Collapsible defaultOpen={isPuzzlesActive} className="group/collapsible">
            <SidebarGroupLabel asChild>
              <CollapsibleTrigger className="flex w-full items-center">
                <Puzzle className="mr-2 size-4" />
                Puzzles
                <Badge variant="outline" className="ml-auto mr-2 text-xs">
                  {completedPuzzleCount}/{puzzles.length}
                </Badge>
                <ChevronDown className="size-4 transition-transform group-data-[state=open]/collapsible:rotate-180" />
              </CollapsibleTrigger>
            </SidebarGroupLabel>
            <CollapsibleContent>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      isActive={pathname === "/puzzles"}
                    >
                      <Link href="/puzzles">
                        <ChevronRight className="size-4" />
                        <span>All Puzzles</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  {puzzles.map((puzzle) => {
                    const isActive = pathname === `/puzzle/${puzzle.id}`;
                    const isComplete = progress.puzzles[puzzle.id]?.completed;

                    return (
                      <SidebarMenuItem key={puzzle.id}>
                        <SidebarMenuButton asChild isActive={isActive}>
                          <Link href={`/puzzle/${puzzle.id}`}>
                            {isComplete ? (
                              <CheckCircle2 className="size-4 text-green-500" />
                            ) : (
                              <ChevronRight className="size-4" />
                            )}
                            <span className="truncate">{puzzle.title}</span>
                          </Link>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    );
                  })}
                </SidebarMenu>
              </SidebarGroupContent>
            </CollapsibleContent>
          </Collapsible>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <div className="p-4 space-y-3">
          {status !== "ready" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{getStatusLabel()}</span>
                <span>{getStatusProgress()}%</span>
              </div>
              <Progress value={getStatusProgress()} className="h-1" />
            </div>
          )}
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              {status === "ready" ? "Python ready" : ""}
            </span>
            <ThemeToggle />
          </div>
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
