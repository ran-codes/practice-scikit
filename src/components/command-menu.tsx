"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Home, Puzzle, Moon, Sun, Play, RotateCcw, BookOpen, GraduationCap, Rocket } from "lucide-react";
import { useTheme } from "next-themes";

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { puzzles } from "@/data/puzzles";
import { PHASE_INFO } from "@/data/types";
import { getExercisesByPhase } from "@/data/course";

interface CommandMenuProps {
  onRun?: () => void;
  onReset?: () => void;
}

const PHASE_ICONS = {
  1: BookOpen,
  2: GraduationCap,
  3: Rocket,
};

export function CommandMenu({ onRun, onReset }: CommandMenuProps) {
  const [open, setOpen] = React.useState(false);
  const router = useRouter();
  const { theme, setTheme } = useTheme();

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const runCommand = React.useCallback((command: () => void) => {
    setOpen(false);
    command();
  }, []);

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        {(onRun || onReset) && (
          <>
            <CommandGroup heading="Actions">
              {onRun && (
                <CommandItem onSelect={() => runCommand(onRun)}>
                  <Play className="mr-2 h-4 w-4" />
                  <span>Run Code</span>
                  <span className="ml-auto text-xs text-muted-foreground">
                    Ctrl+Enter
                  </span>
                </CommandItem>
              )}
              {onReset && (
                <CommandItem onSelect={() => runCommand(onReset)}>
                  <RotateCcw className="mr-2 h-4 w-4" />
                  <span>Reset Code</span>
                  <span className="ml-auto text-xs text-muted-foreground">
                    Ctrl+Shift+R
                  </span>
                </CommandItem>
              )}
            </CommandGroup>
            <CommandSeparator />
          </>
        )}

        <CommandGroup heading="Navigation">
          <CommandItem onSelect={() => runCommand(() => router.push("/"))}>
            <Home className="mr-2 h-4 w-4" />
            <span>Home</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => router.push("/course"))}>
            <BookOpen className="mr-2 h-4 w-4" />
            <span>Course Overview</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => router.push("/puzzles"))}>
            <Puzzle className="mr-2 h-4 w-4" />
            <span>All Puzzles</span>
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        {/* Course Phases */}
        {([1, 2, 3] as const).map((phase) => {
          const exercises = getExercisesByPhase(phase);
          const info = PHASE_INFO[phase];
          const Icon = PHASE_ICONS[phase];

          return (
            <React.Fragment key={phase}>
              <CommandGroup heading={`Phase ${phase}: ${info.title}`}>
                <CommandItem
                  onSelect={() =>
                    runCommand(() => router.push(`/course/${phase}`))
                  }
                >
                  <Icon className="mr-2 h-4 w-4" />
                  <span>View All Phase {phase} Exercises</span>
                </CommandItem>
                {exercises.slice(0, 5).map((exercise) => (
                  <CommandItem
                    key={exercise.id}
                    onSelect={() =>
                      runCommand(() =>
                        router.push(`/course/${phase}/${exercise.id}`)
                      )
                    }
                  >
                    <span className="mr-2 text-muted-foreground text-xs">
                      {exercise.order}.
                    </span>
                    <span>{exercise.title}</span>
                    <span className="ml-auto text-xs text-muted-foreground capitalize">
                      {exercise.difficulty}
                    </span>
                  </CommandItem>
                ))}
                {exercises.length > 5 && (
                  <CommandItem
                    onSelect={() =>
                      runCommand(() => router.push(`/course/${phase}`))
                    }
                  >
                    <span className="text-muted-foreground">
                      +{exercises.length - 5} more exercises...
                    </span>
                  </CommandItem>
                )}
              </CommandGroup>
              <CommandSeparator />
            </React.Fragment>
          );
        })}

        <CommandGroup heading="Puzzles">
          {puzzles.map((puzzle) => (
            <CommandItem
              key={puzzle.id}
              onSelect={() =>
                runCommand(() => router.push(`/puzzle/${puzzle.id}`))
              }
            >
              <Puzzle className="mr-2 h-4 w-4" />
              <span>{puzzle.title}</span>
              {puzzle.difficulty && (
                <span className="ml-auto text-xs text-muted-foreground capitalize">
                  {puzzle.difficulty}
                </span>
              )}
            </CommandItem>
          ))}
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Theme">
          <CommandItem
            onSelect={() =>
              runCommand(() => setTheme(theme === "dark" ? "light" : "dark"))
            }
          >
            {theme === "dark" ? (
              <Sun className="mr-2 h-4 w-4" />
            ) : (
              <Moon className="mr-2 h-4 w-4" />
            )}
            <span>Toggle {theme === "dark" ? "Light" : "Dark"} Mode</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
