import "./App.css";
import { Thread } from "@/components/thread";
import { SignedIn, SignedOut, SignIn, SignUp } from "@clerk/clerk-react";
import { CorvusSVG } from "@/components/icons/corvus";
import { Route, Routes, Navigate } from "react-router-dom";
import { Search, BookOpen, Sparkles } from "lucide-react";
import { StreamProvider } from "./providers/Stream.tsx";
import { ThreadProvider } from "./providers/Thread.tsx";
import { PaperSelectionProvider } from "./providers/PaperSelection.tsx";

const features = [
  {
    icon: Search,
    title: "Semantic search",
    description: "Find relevant papers by meaning, not just keywords.",
  },
  {
    icon: Sparkles,
    title: "AI-powered analysis",
    description: "Get instant summaries and cross-paper insights from your library.",
  },
  {
    icon: BookOpen,
    title: "Curated library",
    description: "Organise your collection and pick up right where you left off.",
  },
];

function LeftPanel() {
  return (
    <div className="hidden lg:flex w-1/2 bg-gray-950 flex-col justify-between p-14 min-h-screen relative z-10 shadow-[20px_0_60px_rgba(0,0,0,0.55)]">
      {/* Top — wordmark */}
      <div className="flex items-center gap-2.5">
        <CorvusSVG width={28} height={28} className="invert" />
        <span className="text-white font-semibold text-lg tracking-tight">Corvus</span>
      </div>

      {/* Centre — headline + features */}
      <div className="space-y-12">
        <div className="space-y-4">
          <h2 className="text-4xl font-semibold text-white tracking-tight leading-snug">
            Research smarter,<br />not harder.
          </h2>
          <p className="text-gray-400 text-base leading-relaxed max-w-sm">
            Corvus is an AI research assistant that helps you discover, analyse,
            and synthesise academic papers — so you can focus on the ideas that
            matter.
          </p>
        </div>

        <div className="space-y-6">
          {features.map(({ icon: Icon, title, description }) => (
            <div key={title} className="flex items-start gap-4">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-gray-800">
                <Icon className="h-4 w-4 text-gray-300" strokeWidth={1.75} />
              </div>
              <div>
                <p className="text-sm font-medium text-white">{title}</p>
                <p className="text-sm text-gray-500 mt-0.5 leading-relaxed">{description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom — fine print */}
      <p className="text-gray-700 text-xs">© {new Date().getFullYear()} Corvus</p>
    </div>
  );
}

function AuthPage({ form }: { form: "sign-in" | "sign-up" }) {
  return (
    <div className="flex min-h-screen">
      <LeftPanel />

      {/* Right panel */}
      <div className="flex flex-1 flex-col items-center justify-center bg-white px-8 py-12">
        {/* Mobile-only header */}
        <div className="mb-8 flex flex-col items-center gap-2 text-center lg:hidden">
          <CorvusSVG width={48} height={48} />
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">Corvus</h1>
          <p className="text-gray-500 text-sm">Your AI research assistant</p>
        </div>

        {form === "sign-in" ? (
          <SignIn routing="path" path="/sign-in" signUpUrl="/sign-up" />
        ) : (
          <SignUp routing="path" path="/sign-up" signInUrl="/sign-in" />
        )}
      </div>
    </div>
  );
}

function AuthenticatedApp() {
  return (
    <ThreadProvider>
      <StreamProvider>
        <PaperSelectionProvider>
          <Thread />
        </PaperSelectionProvider>
      </StreamProvider>
    </ThreadProvider>
  );
}

function App() {
  return (
    <Routes>
      <Route
        path="/sign-in/*"
        element={
          <SignedOut>
            <AuthPage form="sign-in" />
          </SignedOut>
        }
      />
      <Route
        path="/sign-up/*"
        element={
          <SignedOut>
            <AuthPage form="sign-up" />
          </SignedOut>
        }
      />
      <Route
        path="/*"
        element={
          <>
            <SignedOut>
              <Navigate to="/sign-in" replace />
            </SignedOut>
            <SignedIn>
              <AuthenticatedApp />
            </SignedIn>
          </>
        }
      />
    </Routes>
  );
}

export default App;
