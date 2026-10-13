import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useEffect, type ReactNode } from "react";
import { ApiError, UNAUTHENTICATED_EVENT, api } from "./api";

export type AuthUser = {
  id: number;
  email: string | null;
  name: string;
  is_admin: boolean;
  is_demo: boolean;
  created_at: string | null;
  last_login_at: string | null;
};

export type AuthConfig = {
  auth_enabled: boolean;
  allow_signup: boolean;
  demo_accounts: boolean;
  demo_ttl_hours: number;
  password_min_length: number;
  password_reset_by_email: boolean;
};

type Me = { user: AuthUser; auth_enabled: boolean; demo_expires_hours: number | null };

type AuthState = {
  status: "loading" | "signed-in" | "signed-out" | "error";
  user: AuthUser | null;
  config: AuthConfig | undefined;
  authEnabled: boolean;
  demoExpiresHours: number | null;
  signOut: () => Promise<void>;
  retry: () => void;
  /** Call after sign-in/up: reloads the user and every data view. */
  signedIn: (user: AuthUser) => void;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const config = useQuery({
    queryKey: ["auth", "config"],
    queryFn: () => api.get<AuthConfig>("/auth/config"),
    staleTime: Infinity,
  });
  const me = useQuery({
    queryKey: ["auth", "me"],
    queryFn: async () => {
      const s = await api.get<Omit<Me, "user"> & { user: AuthUser | null }>("/auth/session");
      return s.user ? (s as Me) : null;
    },
    staleTime: 5 * 60_000,
    retry: (n, e) => !(e instanceof ApiError && e.status < 500) && n < 2,
  });

  // Any API call that finds the session gone (expired, revoked elsewhere) drops back to sign-in.
  useEffect(() => {
    const onExpired = () => qc.setQueryData(["auth", "me"], null);
    window.addEventListener(UNAUTHENTICATED_EVENT, onExpired);
    return () => window.removeEventListener(UNAUTHENTICATED_EVENT, onExpired);
  }, [qc]);

  const logout = useMutation({
    mutationFn: () => api.post("/auth/logout"),
    onSettled: () => {
      qc.clear();
      qc.setQueryData(["auth", "me"], null);
    },
  });

  const value: AuthState = {
    status: me.isLoading ? "loading" : me.data ? "signed-in" : me.isError ? "error" : "signed-out",
    user: me.data?.user ?? null,
    config: config.data,
    authEnabled: me.data?.auth_enabled ?? config.data?.auth_enabled ?? true,
    demoExpiresHours: me.data?.demo_expires_hours ?? null,
    retry: () => void me.refetch(),
    signOut: async () => {
      await logout.mutateAsync().catch(() => undefined);
    },
    signedIn: (user) => {
      qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "auth" });
      qc.setQueryData<Me>(["auth", "me"], { user, auth_enabled: true, demo_expires_hours: null });
      void qc.invalidateQueries({ queryKey: ["auth", "me"] });
    },
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
