"use server";

import { DashboardClient } from "~/components/dashboard-client";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

export default async function DashboardPage() {
  const session = await auth();

  if (!session?.user?.id) {
    throw new Error("User not authenticated");
  }

  const userId = session.user.id;

  const userData = await db.user.findUniqueOrThrow({
    where: { id: userId },
    select: {
      uploadedFiles: {
        where: {
          uploaded: true,
        },
        select: {
          id: true,
          s3Key: true,
          displayName: true,
          status: true,
          createdAt: true,
          _count: {
            select: {
              clips: true,
            },
          },
        },
      },
      clips: {
        orderBy: {
          createdAt: "desc",
        },
      },
    },
  });

  const formattedFiles = userData.uploadedFiles.map((file) => ({
    id: file.id,
    s3Key: file.s3Key,
    filename: file.displayName ?? "Unkown filename",
    status: file.status,
    clipsCount: file._count.clips,
    createdAt: file.createdAt,
  }));

  return (
    <DashboardClient uploadedFiles={formattedFiles} clips={userData.clips} />
  );
}