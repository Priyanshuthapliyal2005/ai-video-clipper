"use client";
import { Button } from "~/components/ui/button";
import { motion } from "framer-motion";
import Link from "next/link";
import {AuroraBackground} from "~/components/ui/aurora-background";
import LiquidEther from "~/components/ui/ether";
import LandingNav from "~/components/landing-nav";

const fadeUpVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export default function App() {
  return (
  <main className="relative z-10 min-h-screen max-w-5xl mx-auto flex flex-col items-center justify-center px-4 md:px-6 lg:px-8 pt-8 pb-12 gap-16">
      <LandingNav />
      <div className="text-center flex flex-col gap-6 max-w-2xl mx-auto">
        {/* import LiquidEther from './LiquidEther'; */}

              {/* LiquidEther runs in a fixed interactive layer above the aurora but behind UI */}
        <div className="text-center flex flex-col gap-2">
          <motion.h1
            className={`text-4xl md:text-5xl font-medium tracking-tight`}
            variants={fadeUpVariants}
            initial="initial"
            animate="animate"
            transition={{ duration: 0.5 }}
          >
            Ship Clips. Make Nerd.
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
        src="/pd.mp4"
      />
      <div className="fixed inset-0 w-full h-full -z-[30] pointer-events-none">
        <AuroraBackground showRadialGradient={false} />
      </div>

  <div className="fixed inset-0 w-full h-full -z-[20] pointer-events-none">
        <div className="w-full h-full">
          <LiquidEther
            colors={[ '#5227FF', '#FF9FFC', '#B19EEF' ]}
            mouseForce={20}
            cursorSize={100}
            isViscous={false}
            viscous={30}
            iterationsViscous={32}
            iterationsPoisson={32}
            resolution={0.5}
            isBounce={false}
            autoDemo={false}
            autoSpeed={0.5}
            autoIntensity={2.2}
            takeoverDuration={0.25}
            autoResumeDelay={3000}
            autoRampDuration={0.6}
          />
        </div>
      </div>


      <footer className="text-sm text-foreground flex items-center gap-2">
        <p>© 2025 Podcast Clipper. All rights reserved.</p>
        <Link href="/terms" className="underline text-foreground">
          Terms & Conditions
        </Link>
      </footer>
    </main>
  );
}