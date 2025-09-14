import "./index.css";
import App from "./App.tsx";
import { createRoot } from "react-dom/client";
import { StreamProvider } from "./providers/Stream.tsx";
import { ThreadProvider } from "./providers/Thread.tsx";
import { Toaster } from "@/components/ui/sonner";
import { NuqsAdapter } from "nuqs/adapters/react-router/v6";
import { BrowserRouter } from "react-router-dom";

createRoot(document.getElementById("root")!).render(
  <BrowserRouter>
    <NuqsAdapter>
      <ThreadProvider>
        <StreamProvider>
          <App />
        </StreamProvider>
      </ThreadProvider>
      <Toaster />
    </NuqsAdapter>
  </BrowserRouter>,
);
