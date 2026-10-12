import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { Toaster } from "sonner";
import { ErrorBoundary, reportClientError } from "./components/ErrorBoundary";
import Layout from "./components/Layout";
import { Skeleton } from "./components/ui";
import "./index.css";
import { lazyWithRetry } from "./lib/lazy";

const Overview = lazyWithRetry(() => import("./pages/Overview"));
const Advisor = lazyWithRetry(() => import("./pages/Advisor"));
const Simulator = lazyWithRetry(() => import("./pages/Simulator"));
const Goals = lazyWithRetry(() => import("./pages/Goals"));
const Insights = lazyWithRetry(() => import("./pages/Insights"));
const Transactions = lazyWithRetry(() => import("./pages/Transactions"));
const Chat = lazyWithRetry(() => import("./pages/Chat"));
const Digests = lazyWithRetry(() => import("./pages/Digests"));
const Agents = lazyWithRetry(() => import("./pages/Agents"));
const Decisions = lazyWithRetry(() => import("./pages/Decisions"));
const Evaluation = lazyWithRetry(() => import("./pages/Evaluation"));
const SettingsPage = lazyWithRetry(() => import("./pages/Settings"));

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 15_000 } },
});

function Fallback() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-64" />
      <div className="grid gap-4 md:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-80" />
    </div>
  );
}

function AppRoutes() {
  const location = useLocation();
  return (
    <Layout>
      <ErrorBoundary resetKey={location.pathname}>
        <Suspense fallback={<Fallback />}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/advisor" element={<Advisor />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/goals" element={<Goals />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/transactions" element={<Transactions />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/digests" element={<Digests />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/decisions" element={<Decisions />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Overview />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </Layout>
  );
}

// Errors outside React's render (event handlers, timers, rejected promises) still reach the server log.
window.addEventListener("error", (e) => reportClientError(e.error ?? e.message, { source: "window.onerror" }));
window.addEventListener("unhandledrejection", (e) => reportClientError(e.reason, { source: "unhandledrejection" }));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
      <Toaster richColors position="top-right" closeButton />
    </QueryClientProvider>
  </StrictMode>,
);
