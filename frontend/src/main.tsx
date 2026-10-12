import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster } from "sonner";
import Layout from "./components/Layout";
import { Skeleton } from "./components/ui";
import "./index.css";

const Overview = lazy(() => import("./pages/Overview"));
const Advisor = lazy(() => import("./pages/Advisor"));
const Simulator = lazy(() => import("./pages/Simulator"));
const Goals = lazy(() => import("./pages/Goals"));
const Insights = lazy(() => import("./pages/Insights"));
const Transactions = lazy(() => import("./pages/Transactions"));
const Chat = lazy(() => import("./pages/Chat"));
const Digests = lazy(() => import("./pages/Digests"));
const Agents = lazy(() => import("./pages/Agents"));
const Decisions = lazy(() => import("./pages/Decisions"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const SettingsPage = lazy(() => import("./pages/Settings"));

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

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
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
        </Layout>
      </BrowserRouter>
      <Toaster richColors position="top-right" closeButton />
    </QueryClientProvider>
  </StrictMode>,
);
