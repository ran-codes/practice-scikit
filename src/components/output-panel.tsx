"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";

interface Props {
  output: { html?: string; error?: string } | null;
  isRunning: boolean;
}

export function OutputPanel({ output, isRunning }: Props) {
  if (isRunning) {
    return (
      <Card>
        <CardContent className="p-4">
          <div className="space-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-20 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!output) {
    return (
      <Card>
        <CardContent className="p-4 text-muted-foreground">
          <p className="italic">Click &quot;Run&quot; to execute your code</p>
        </CardContent>
      </Card>
    );
  }

  if (output.error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          <pre className="mt-2 text-sm whitespace-pre-wrap font-mono overflow-x-auto">
            {output.error}
          </pre>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Card>
      <CardContent className="p-4 overflow-x-auto">
        <div
          className="sklearn-output"
          dangerouslySetInnerHTML={{ __html: output.html || "" }}
        />
      </CardContent>
    </Card>
  );
}
