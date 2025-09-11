"use client";

import Link from "next/link";
import { Button } from "./ui/button";

export default function LandingNav() {
  return (
    <header className="w-full sticky top-0 z-20 flex justify-center border-b bg-transparent">
      <div className="container mx-auto flex items-center justify-between px-4 py-4">
        <Link href="/" className="flex items-center gap-1 text-xl font-medium tracking-tight">
          <span className="text-foreground">Podcast</span>
          <span className="text-foreground font-light"> Clipper</span>
        </Link>

        <div className="flex items-center gap-3">
          <Link href="/login" className="text-sm text-muted-foreground underline">
            Sign in
          </Link>

          <Link href="/dashboard">
            <Button size="sm">Get started</Button>
          </Link>
        </div>
      </div>

    </header>
  );
}
