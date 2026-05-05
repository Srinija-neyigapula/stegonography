import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/lib/auth-context";
import Nav from "@/components/Nav";

export default function ProtectedShell({ children }) {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (loading) return;
    if (!user) {
      navigate("/login", { replace: true });
    } else {
      setShow(true);
    }
  }, [user, loading, navigate]);

  if (loading || !show) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-white flex items-center justify-center font-mono text-sm">
        <div className="flex items-center gap-3">
          <span className="inline-block w-2 h-2 bg-[#FF3B30] rounded-full animate-pulse" />
          loading lab…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white">
      <Nav />
      {children}
    </div>
  );
}
