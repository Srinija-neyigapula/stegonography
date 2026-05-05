import { useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { useAuth } from "@/lib/auth-context";

const LOGIN_BG =
  "https://static.prod-images.emergentagent.com/jobs/e919bdbe-c645-4b91-8ad4-f67c81ad43c8/images/0de0e8e12324679ac0e2fd75a7c0090fb4cfc2b1cc25a77d3ed76895aaaa20fc.png";

export default function Login() {
  const navigate = useNavigate();
  const { user, loading } = useAuth();

  useEffect(() => {
    if (!loading && user) navigate("/sender", { replace: true });
  }, [user, loading, navigate]);

  const handleLogin = () => {
    // REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
    const redirectUrl = window.location.origin + "/sender";
    window.location.href = `https://auth.emergentagent.com/?redirect=${encodeURIComponent(redirectUrl)}`;
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0A0A0A] text-white">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-40"
        style={{ backgroundImage: `url(${LOGIN_BG})` }}
      />
      <div className="absolute inset-0 bg-gradient-to-br from-[#0A0A0A]/95 via-[#0A0A0A]/80 to-[#0A0A0A]/95 backdrop-blur-sm" />

      <div className="relative z-10 min-h-screen flex flex-col">
        <header className="px-6 md:px-12 py-6 flex items-center justify-between border-b border-white/5">
          <div className="flex items-center gap-3 font-mono">
            <div className="w-8 h-8 border border-[#FF3B30] flex items-center justify-center">
              <div className="w-2 h-2 bg-[#FF3B30] animate-pulse" />
            </div>
            <span className="text-sm tracking-[0.3em] text-white">SGIC</span>
            <span className="text-xs text-gray-500 hidden md:inline">
              // SECURE_GENERATIVE_IMAGE_COMMUNICATION
            </span>
          </div>
          <span className="font-mono text-xs text-gray-500">v0.1 · research_demo</span>
        </header>

        <main className="flex-1 flex items-center justify-center px-6 py-12">
          <div className="grid md:grid-cols-2 gap-12 max-w-6xl w-full items-center">
            <div className="space-y-8">
              <div className="font-mono text-xs text-[#FF3B30] tracking-[0.4em]">
                / RESEARCH DEMO_
              </div>
              <h1 className="font-mono text-4xl md:text-6xl font-bold leading-tight tracking-tight">
                Hidden in
                <br />
                <span className="text-[#FF3B30]">plain pixels.</span>
              </h1>
              <p className="text-gray-400 leading-relaxed max-w-md font-sans">
                A reference implementation for{" "}
                <span className="text-white">Secure Generative Image
                Communication</span> — embedding RSA-encrypted, zlib-compressed
                payloads inside cover images via key-seeded LSB diffusion. End-to-end
                round-trip with PSNR / SSIM / Chi-Square / RS analysis.
              </p>

              <ul className="space-y-2 font-mono text-xs text-gray-500">
                {[
                  "[01] payload  → zlib · RSA-OAEP · stream-cipher",
                  "[02] embed    → key-seeded LSB · diffusion priors",
                  "[03] verify   → PSNR · SSIM · BPP · χ² · RS",
                  "[04] attack   → noise · jpeg · resize",
                ].map((row) => (
                  <li key={row} className="flex items-center gap-3">
                    <span className="w-1 h-1 bg-[#FF3B30]" />
                    {row}
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-[#141414] border border-white/10 p-8 md:p-10 space-y-8">
              <div className="space-y-2">
                <div className="font-mono text-xs text-gray-500 tracking-[0.3em]">
                  / AUTHENTICATE
                </div>
                <h2 className="font-mono text-2xl text-white">
                  Sign in to access the lab
                </h2>
                <p className="text-sm text-gray-400">
                  Researcher access only. We use Google for one-click sign-in via
                  Emergent's managed auth.
                </p>
              </div>

              <button
                onClick={handleLogin}
                data-testid="google-login-button"
                className="w-full bg-white hover:bg-gray-100 text-black font-mono text-sm py-3.5 px-6 rounded-sm transition-colors flex items-center justify-center gap-3 group"
              >
                <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
                  <path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615z"/>
                  <path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"/>
                  <path fill="#FBBC05" d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.997 8.997 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z"/>
                  <path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z"/>
                </svg>
                <span className="tracking-wider">CONTINUE WITH GOOGLE</span>
                <span className="text-[#FF3B30] opacity-0 group-hover:opacity-100 transition-opacity">→</span>
              </button>

              <div className="border-t border-white/5 pt-6 font-mono text-[10px] text-gray-600 leading-relaxed">
                BY_CONTINUING YOU AGREE TO STORE A SESSION COOKIE FOR 7 DAYS.
                <br />
                NO PASSWORDS. NO TRACKING BEYOND PROFILE.
              </div>
            </div>
          </div>
        </main>

        <footer className="px-6 md:px-12 py-4 border-t border-white/5 font-mono text-[10px] text-gray-600 flex justify-between">
          <span>© SGIC LAB / 2026</span>
          <span>STATUS: <span className="text-[#32D74B]">● ONLINE</span></span>
        </footer>
      </div>
    </div>
  );
}
