import { notFound } from "next/navigation";
import { ExercisePage } from "@/components/exercise-page";
import { getCourseExerciseById, generateCourseStaticParams } from "@/data/course";

interface Props {
  params: Promise<{
    phase: string;
    id: string;
  }>;
}

export default async function CourseExercisePage({ params }: Props) {
  const { id, phase } = await params;
  const exercise = getCourseExerciseById(id);

  if (!exercise || exercise.phase !== parseInt(phase)) {
    notFound();
  }

  return <ExercisePage exercise={exercise} />;
}

export function generateStaticParams() {
  return generateCourseStaticParams();
}
