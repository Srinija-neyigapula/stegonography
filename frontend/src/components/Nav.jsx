import { Link, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "@/lib/auth-context";

export default function Nav() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const onLogout = async () => {
    await logout();
    navigate("/login", { replace: true });
  };

  const navClass = ({ isActive }) =>
    `font-mono text-xs tracking-[0.3em] px-3 py-1.5 border transition-colors ${
      isActive
        ? "border-[#FF3B30] text-[#FF3B30] bg-[#FF3B30]/5"
        : "border-transparent text-gray-400 hover:text-white hover:border-white/20"
    }`;

  return (
    <header className="sticky top-0 z-40 bg-[#0A0A0A]/95 backdrop-blur border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        <Link to="/sender" className="flex items-center gap-3">
          <div className="w-7 h-7 border border-[#FF3B30] flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-[#FF3B30] animate-pulse" />
          </div>
          <span className="font-mono text-xs tracking-[0.3em] text-white">SGIC</span>
          <span className="font-mono text-[10px] text-gray-600 hidden md:inline">
            / SECURE_GEN_IMG_COMM
          </span>
        </Link>

        <nav className="flex items-center gap-2">
          <NavLink to="/sender" className={navClass} data-testid="nav-sender">
            SENDER
          </NavLink>
          <NavLink to="/receiver" className={navClass} data-testid="nav-receiver">
            RECEIVER
          </NavLink>
        </nav>

        <div className="flex items-center gap-3">
          {user?.picture && (
            <img
              src={user.picture}
              alt={user.name}
              className="w-7 h-7 rounded-full border border-white/10"
            />
          )}
          <span className="font-mono text-xs text-gray-400 hidden md:inline">
            {user?.email}
          </span>
          <button
            onClick={onLogout}
            data-testid="logout-button"
            className="font-mono text-xs tracking-[0.2em] text-gray-500 hover:text-[#FF3B30] transition-colors"
          >
            LOGOUT
          </button>
        </div>
      </div>
    </header>
  );
}
