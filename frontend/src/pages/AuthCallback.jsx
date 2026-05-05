import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { auth } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

// REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
export default function AuthCallback() {
  const navigate = useNavigate();
  const { setUser } = useAuth();
  const hasProcessed = useRef(false);

  useEffect(() => {
    if (hasProcessed.current) return;
    hasProcessed.current = true;

    const hash = window.location.hash || "";
    const params = new URLSearchParams(hash.replace(/^#/, ""));
    const sid = params.get("session_id");
    if (!sid) {
      navigate("/login", { replace: true });
      return;
    }
    auth
      .session(sid)
      .then((data) => {
        setUser(data.user);
        navigate("/sender", { replace: true, state: { user: data.user } });
      })
      .catch(() => navigate("/login", { replace: true }));
  }, [navigate, setUser]);

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white flex items-center justify-center font-mono text-sm">
      <div className="flex items-center gap-3">
        <span className="inline-block w-2 h-2 bg-[#FF3B30] rounded-full animate-pulse" />
        Establishing secure session…
      </div>
    </div>
  );
}
