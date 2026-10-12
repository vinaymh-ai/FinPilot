import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "./api";
import type { Category, Meta } from "./types";
import { setCurrency } from "./format";

export function useMeta() {
  return useQuery({
    queryKey: ["meta"],
    queryFn: async () => {
      const m = await api.get<Meta>("/meta");
      setCurrency(m.currency);
      return m;
    },
    staleTime: 60_000,
  });
}

export function useCategories(): Record<string, Category> {
  const { data } = useMeta();
  const out: Record<string, Category> = {};
  for (const c of data?.categories ?? []) out[c.id] = c;
  return out;
}

/** Invalidate every data view after the agents have produced new outputs. */
export function useRefreshAll() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries();
}

export function useRunAgents() {
  const refresh = useRefreshAll();
  return useMutation({
    mutationFn: (body: { recategorize?: boolean; deliver_digest?: boolean } = {}) =>
      api.post<{ run_id: number; status: string; summary: Record<string, any> }>("/runs", body),
    onSuccess: (r) => {
      toast.success(`Agents finished — run #${r.run_id}`, {
        description: r.summary?.headline ?? `${r.summary?.recommendations ?? 0} recommendations`,
      });
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
