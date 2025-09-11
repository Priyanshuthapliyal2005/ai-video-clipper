"use client";
import { Button } from "~/components/ui/button";
import { motion } from "framer-motion";
import Link from "next/link";
import RetroGrid from "~/components/ui/retro-grid";

const fadeUpVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export default function App() {
  return (
    <main className="min-h-screen max-w-5xl mx-auto flex flex-col items-center justify-center px-4 md:px-6 lg:px-8 pt-32 pb-12 gap-16">
      <div className="text-center flex flex-col gap-6 max-w-2xl mx-auto">
        <div className="text-center flex flex-col gap-2">
          <motion.h1
            className={`text-4xl md:text-5xl font-medium tracking-tight`}
            variants={fadeUpVariants}
            initial="initial"
            animate="animate"
            transition={{ duration: 0.5 }}
          >
            Snip Clips. Make Bangers.
          </motion.h1>

          <motion.p
            className="text-secondary-foreground text-lg max-w-md mx-auto"
            variants={fadeUpVariants}
            initial="initial"
            animate="animate"
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            Podcast Clipper is your platform for YT clips. Create bangers moments from Podcasts.
          </motion.p>
        </div>

        <motion.div
          className="flex gap-2 items-center justify-center"
          variants={fadeUpVariants}
          initial="initial"
          animate="animate"
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Link href="/dashboard">
            <Button size="lg">Get started</Button>
          </Link>
        </motion.div>
      </div>
      
      <video
        autoPlay
        muted
        loop
        playsInline
        className="border rounded-3xl"
        src="/clippa.mp4"
      />
      <div className="fixed inset-0 -z-10 w-full h-full pointer-events-none">
        <RetroGrid className="w-full h-full" />
      </div>


      <footer className="text-sm text-muted-foreground flex items-center gap-2">
        <p>© 2025 Podcast Clipper. All rights reserved.</p>
        <Link href="/terms" className="underline">
          Terms & Conditions
        </Link>
      </footer>
    </main>
  );
}