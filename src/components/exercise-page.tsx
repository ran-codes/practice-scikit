"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Play, RotateCcw, ChevronLeft, ChevronRight, Check, Lightbulb, Eye, ExternalLink } from "lucide-react";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { usePyodide } from "@/contexts/pyodide-context";
import { useProgress } from "@/contexts/progress-context";
import { OutputPanel } from "@/components/output-panel";
import type { CourseExercise } from "@/data/types";
import { getNextExercise, getPreviousExercise } from "@/data/course";

// Dynamic import for CodeMirror (client-only)
const CodeEditor = dynamic(
  () => import("@/components/code-editor").then((m) => m.CodeEditor),
  {
    ssr: false,
    loading: () => <Skeleton className="h-full w-full rounded-lg" />,
  }
);

interface Props {
  exercise: CourseExercise;
}

export function ExercisePage({ exercise }: Props) {
  const { runPython, status, resetNamespace } = usePyodide();
  const { isComplete, markComplete, getSavedCode, saveCode } = useProgress();
  const [code, setCode] = useState(getSavedCode("course", exercise.id) || exercise.code);
  const [output, setOutput] = useState<{
    html?: string;
    error?: string;
  } | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("problem");
  const [hintsOpen, setHintsOpen] = useState(false);
  const [showSolutionDialog, setShowSolutionDialog] = useState(false);

  const isReady = status === "ready";
  const hasSolution = !!exercise.solution;
  const isLoading = status === "loading-runtime" || status === "loading-packages";
  const completed = isComplete("course", exercise.id);

  const prevExercise = getPreviousExercise(exercise.id);
  const nextExercise = getNextExercise(exercise.id);

  // Reset Python namespace when switching exercises
  useEffect(() => {
    if (status === "ready") {
      resetNamespace();
    }
  }, [exercise.id, status, resetNamespace]);

  // Save code changes
  useEffect(() => {
    const timeout = setTimeout(() => {
      saveCode("course", exercise.id, code);
    }, 1000);
    return () => clearTimeout(timeout);
  }, [code, exercise.id, saveCode]);

  const handleRun = useCallback(async () => {
    if (!isReady || isRunning) return;

    setIsRunning(true);
    setOutput(null);
    setActiveTab("output");

    const result = await runPython(code);
    setOutput(result.success ? { html: result.html } : { error: result.error });

    setIsRunning(false);
  }, [isReady, isRunning, runPython, code]);

  const handleReset = useCallback(() => {
    setCode(exercise.code);
    setOutput(null);
    setActiveTab("problem");
  }, [exercise.code]);

  const handleMarkComplete = useCallback(() => {
    markComplete("course", exercise.id, code);
  }, [markComplete, exercise.id, code]);

  const handleSolutionClick = useCallback(() => {
    // Check if user has modified the code from the original
    const hasModified = code !== exercise.code;
    if (hasModified) {
      setShowSolutionDialog(true);
    } else {
      // No modifications, just load solution
      if (exercise.solution) {
        setCode(exercise.solution);
      }
    }
  }, [code, exercise.code, exercise.solution]);

  const handleConfirmSolution = useCallback(() => {
    if (exercise.solution) {
      setCode(exercise.solution);
    }
    setShowSolutionDialog(false);
  }, [exercise.solution]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleRun();
      }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "R") {
        e.preventDefault();
        handleReset();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleRun, handleReset]);

  return (
    <div className="h-[calc(100vh-2rem)] p-4">
      <ResizablePanelGroup direction="horizontal" className="h-full rounded-lg border">
        {/* Left Panel - Code Editor */}
        <ResizablePanel defaultSize={55} minSize={30}>
          <div className="flex flex-col h-full">
            {/* Editor Toolbar */}
            <div className="flex items-center justify-between p-3 border-b bg-muted/30">
              <div className="flex items-center gap-3">
                <Link href={`/course/${exercise.phase}`}>
                  <Button variant="ghost" size="sm" className="h-8 px-2">
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                </Link>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-sm">{exercise.order}.</span>
                  <h1 className="font-semibold truncate">{exercise.title}</h1>
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
                {completed && (
                  <Badge variant="outline" className="text-green-600 border-green-600">
                    <Check className="h-3 w-3 mr-1" />
                    Complete
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      onClick={handleRun}
                      disabled={!isReady || isRunning}
                    >
                      <Play className="h-4 w-4 mr-1" />
                      {isRunning ? "Running..." : isLoading ? "Loading..." : "Run"}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Run code (Ctrl+Enter)</p>
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleReset}
                      disabled={isRunning}
                    >
                      <RotateCcw className="h-4 w-4 mr-1" />
                      Reset
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Reset code (Ctrl+Shift+R)</p>
                  </TooltipContent>
                </Tooltip>
                {hasSolution && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleSolutionClick}
                        disabled={isRunning}
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        Solution
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Load solution code</p>
                    </TooltipContent>
                  </Tooltip>
                )}
                {!completed && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleMarkComplete}
                      >
                        <Check className="h-4 w-4 mr-1" />
                        Done
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Mark as complete</p>
                    </TooltipContent>
                  </Tooltip>
                )}
              </div>
            </div>

            {/* Editor */}
            <div className="flex-1 overflow-hidden">
              <CodeEditor value={code} onChange={setCode} />
            </div>

            {/* Navigation Footer */}
            <div className="flex items-center justify-between p-2 border-t bg-muted/30">
              {prevExercise ? (
                <Link href={`/course/${prevExercise.phase}/${prevExercise.id}`}>
                  <Button variant="ghost" size="sm">
                    <ChevronLeft className="h-4 w-4 mr-1" />
                    {prevExercise.title}
                  </Button>
                </Link>
              ) : (
                <div />
              )}
              {nextExercise ? (
                <Link href={`/course/${nextExercise.phase}/${nextExercise.id}`}>
                  <Button variant="ghost" size="sm">
                    {nextExercise.title}
                    <ChevronRight className="h-4 w-4 ml-1" />
                  </Button>
                </Link>
              ) : (
                <Link href="/course">
                  <Button variant="ghost" size="sm">
                    Back to Course
                    <ChevronRight className="h-4 w-4 ml-1" />
                  </Button>
                </Link>
              )}
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        {/* Right Panel - Problem & Output */}
        <ResizablePanel defaultSize={45} minSize={25}>
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="h-full flex flex-col"
          >
            <div className="border-b bg-muted/30">
              <TabsList className="w-full justify-start rounded-none border-none bg-transparent h-auto p-0">
                <TabsTrigger
                  value="problem"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-4 py-3"
                >
                  Problem
                </TabsTrigger>
                <TabsTrigger
                  value="output"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-4 py-3"
                >
                  Output
                  {output && (
                    <span
                      className={`ml-2 w-2 h-2 rounded-full ${
                        output.error ? "bg-destructive" : "bg-green-500"
                      }`}
                    />
                  )}
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="problem" className="flex-1 overflow-auto m-0 p-4">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <h2 className="text-lg font-semibold mb-2">Task</h2>
                <p className="text-muted-foreground">{exercise.description}</p>

                {exercise.hints && exercise.hints.length > 0 && (
                  <Collapsible open={hintsOpen} onOpenChange={setHintsOpen} className="mt-4">
                    <CollapsibleTrigger asChild>
                      <Button variant="outline" size="sm" className="w-full justify-start">
                        <Lightbulb className="h-4 w-4 mr-2" />
                        {hintsOpen ? "Hide Hints" : "Show Hints"}
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-2">
                      <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground bg-muted/50 p-3 rounded-md">
                        {exercise.hints.map((hint, i) => (
                          <li key={i}>{hint}</li>
                        ))}
                      </ul>
                    </CollapsibleContent>
                  </Collapsible>
                )}

                <h3 className="text-md font-semibold mt-6 mb-2">Instructions</h3>
                <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Read through the code and understand the setup</li>
                  <li>Complete the TODO sections in the code</li>
                  <li>Click Run or press Ctrl+Enter to execute</li>
                  <li>Click Done when finished</li>
                </ol>

                {exercise.concepts && exercise.concepts.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-md font-semibold mb-2">Concepts</h3>
                    <div className="flex flex-wrap gap-2">
                      {exercise.concepts.map((concept) => (
                        <Badge key={concept} variant="outline" className="text-xs">
                          {concept}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {exercise.docsUrl && exercise.docsUrl.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-md font-semibold mb-2">Documentation</h3>
                    <ul className="space-y-1">
                      {exercise.docsUrl.map((url, i) => (
                        <li key={i}>
                          <a
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm text-primary hover:underline flex items-center gap-1"
                          >
                            <ExternalLink className="h-3 w-3" />
                            {url.split('/stable/')[1]?.replace('.html', '').replace(/#/g, ' > ') || 'sklearn docs'}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="output" className="flex-1 overflow-auto m-0 p-4">
              <OutputPanel output={output} isRunning={isRunning} />
            </TabsContent>
          </Tabs>
        </ResizablePanel>
      </ResizablePanelGroup>

      {/* Solution Confirmation Dialog */}
      <Dialog open={showSolutionDialog} onOpenChange={setShowSolutionDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Load Solution?</DialogTitle>
            <DialogDescription>
              You have made changes to the code. Loading the solution will
              replace your current code. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSolutionDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleConfirmSolution}>
              Load Solution
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
