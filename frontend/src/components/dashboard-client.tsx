"use client";

import type { Clip } from "@prisma/client";
import Link from "next/link";
import { Button } from "./ui/button";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";
import { TabsContent } from "@radix-ui/react-tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import Dropzone, {
  type DropzoneState,
  type DropzoneProps,
} from "shadcn-dropzone";
import { Loader2, UploadCloud } from "lucide-react";
import { useState } from "react";
import { generateUploadUrl } from "~/action/s3";
import { toast } from "sonner";
import { processVideo } from "~/action/generation";

export function DashboardClient({
  uploadedFiles,
  clips,
}: {
  uploadedFiles: {
    id: string;
    s3Key: string;
    filename: string;
    status: string;
    clipCount: number;
    createdAt: Date;
  }[];
  clips: Clip[];
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);

  const handleDrop = (acceptedFiles: File[]) => {
    setFiles(acceptedFiles);
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      return;
    }

    const file = files[0]!;
    setUploading(true);

    try {
      const { success, signedUrl, uploadedFileId } = await generateUploadUrl({
        filename: file.name,
        contentType: file.type,
      });

      if (!success) throw new Error("Failed to generate upload URL");

      const uploadedResponse = await fetch(signedUrl, {
        method: "PUT",
        body: file,
        headers: {
          "Content-Type": file.type,
        },
      });

      if (!uploadedResponse.ok)
        throw new Error(
          `Upload files with status : ${uploadedResponse.status}`,
        );

      await processVideo(uploadedFileId);
      

      setFiles([]);

      toast.success("Video uploaded successfully", {
        description:
          "Your video has been scheduled for processing. Check the status below.",
        duration: 5000,
      });
    } catch (error) {
      toast.error("Upload failed", {
        description:
          "There was a problem uploading your video. Please try again.",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col space-y-6 px-4 py-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Podcast Clipper
          </h1>
          <p className="text-muted-foreground">
            Upload your podcast and get AI-generated clips instantly
          </p>
        </div>
        <Link href="/dashbaord/billing">
          <Button>Buy Credits</Button>
        </Link>
      </div>
      <Tabs defaultValue="upload">
        <TabsList>
          <TabsTrigger value="upload">Upload</TabsTrigger>
          <TabsTrigger value="my-clips">My-Clips</TabsTrigger>
        </TabsList>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload Podcast</CardTitle>
              <CardDescription>
                Upload your audio or video file to generate clips.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Dropzone
                onDrop={handleDrop}
                accept={{ "video/mp4": [".mp4"] }}
                maxSize={500 * 1024 * 1024}
                disabled={uploading}
              >
                {(dropzone: DropzoneState) => (
                  <>
                    <div className="flex flex-col items-center justify-center space-y-4 rounded-lg p-10 text-center">
                      <UploadCloud className="text-muted-foreground h-12 w-12" />
                      <p className="font-medium">Drag and drop your file</p>
                      <p className="text-muted-foreground text-sm">
                        or Click to browse (Mp4 up to 500MB)
                      </p>
                      <Button
                        className="cursor-pointer"
                        variant="default"
                        size="sm"
                        disabled={uploading}
                      >
                        Select File
                      </Button>
                    </div>
                  </>
                )}
              </Dropzone>

              <div className="flex items-start justify-between">
                <div>
                  {files.length > 0 && (
                    <div className="space-y-1 text-sm">
                      <p className="font-medium">Selected File: </p>
                      {files.map((file) => (
                        <p className="text-muted-foreground">{file.name}</p>
                      ))}
                    </div>
                  )}
                </div>
                <Button
                  disabled={files.length === 0 || uploading}
                  onClick={handleUpload}
                >
                  {uploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    "Upload and Generate Clips"
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="upload"></TabsContent>
      </Tabs>
    </div>
  );
}
