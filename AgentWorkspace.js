import React, { useState } from 'react';

export default function AgentWorkspace() {
  const [permissionMode, setPermissionMode] = useState('ask'); // ask, plan, auto

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-slate-100 font-sans overflow-hidden">
      
      {/* 1. TOP NAVIGATION BAR */}
      <header className="h-12 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between px-4 select-none">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-indigo-500 animate-pulse" />
          <span className="font-semibold text-sm tracking-wide text-slate-200">AGENT.IDE</span>
          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">v1.0.0</span>
        </div>
        
        {/* Active Session Tabs */}
        <div className="flex gap-1 h-full items-end">
          <button type="button" className="px-4 py-2 text-xs font-medium border-t-2 border-indigo-500 bg-slate-900 text-slate-100 rounded-t-md">
            ⚡ main-agent-task
          </button>
          <button type="button" className="px-4 py-2 text-xs font-medium border-t-2 border-transparent text-slate-400 hover:bg-slate-900/50 rounded-t-md">
            ↳ fix-auth-bug
          </button>
        </div>

        <div className="text-xs text-slate-500">Workspace: /users/project/root</div>
      </header>

      {/* MAIN WORKSPACE SPLIT */}
      <main className="flex-1 flex overflow-hidden">
        
        {/* 2. LEFT SIDEBAR: AGENT CHAT & TERMINAL CANVAS */}
        <section className="w-1/2 flex flex-col border-r border-slate-800 bg-slate-900/20">
          {/* Terminal Headers / Controls */}
          <div className="p-3 border-b border-slate-800 flex items-center justify-between bg-slate-900/40">
            <span className="text-xs font-mono text-slate-400">Agent Stream (bash)</span>
            
            {/* Mode Selector Matrix */}
            <div className="flex bg-slate-950 p-1 rounded-md border border-slate-800 gap-1">
              <button 
                type="button"
                onClick={() => setPermissionMode('plan')}
                className={`px-2 py-1 text-[11px] rounded font-medium transition-all ${permissionMode === 'plan' ? 'bg-slate-800 text-indigo-400 shadow-sm' : 'text-slate-500'}`}
              >
                Plan Only
              </button>
              <button 
                type="button"
                onClick={() => setPermissionMode('ask')}
                className={`px-2 py-1 text-[11px] rounded font-medium transition-all ${permissionMode === 'ask' ? 'bg-slate-800 text-amber-400 shadow-sm' : 'text-slate-500'}`}
              >
                Ask First
              </button>
              <button 
                type="button"
                onClick={() => setPermissionMode('auto')}
                className={`px-2 py-1 text-[11px] rounded font-medium transition-all ${permissionMode === 'auto' ? 'bg-slate-800 text-emerald-400 shadow-sm' : 'text-slate-500'}`}
              >
                Auto-Run
              </button>
            </div>
          </div>

          {/* Terminal Streaming Output */}
          <div className="flex-1 p-4 font-mono text-xs overflow-y-auto space-y-3 bg-slate-950/40">
            <div className="text-slate-500">$ claude-code run test</div>
            <div className="text-emerald-400">✔ Found 12 source files to index.</div>
            <div className="text-slate-300">Agent: "I am going to inspect your configuration and update the outdated dependencies."</div>
            <div className="p-3 bg-slate-900/60 rounded border border-slate-800 text-indigo-300 flex items-center justify-between">
              <span>🔧 Tool Call: modifying `package.json`...</span>
              <span className="text-[10px] bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded border border-indigo-500/20">Pending Approval</span>
            </div>
          </div>

          {/* Interactive Prompter Input */}
          <div className="p-4 border-t border-slate-800 bg-slate-900/30">
            <div className="relative flex items-center bg-slate-950 rounded-lg border border-slate-800 focus-within:border-indigo-500/50 p-2">
              <input 
                type="text" 
                placeholder="Ask the agent to build, refactor, or run diagnostics..." 
                className="w-full bg-transparent px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none"
              />
              <button type="button" className="absolute right-2 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-medium transition-all">
                Execute
              </button>
            </div>
          </div>
        </section>

        {/* 3. RIGHT SIDEBAR: REACTIVE DIFFS & PREVIEWS */}
        <section className="w-1/2 flex flex-col bg-slate-950">
          <div className="h-10 border-b border-slate-800 flex items-center px-4 bg-slate-900/10 justify-between">
            <span className="text-xs font-medium text-slate-400">Proposed File Changes</span>
            <div className="flex gap-2">
              <button type="button" className="px-2.5 py-1 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/30 rounded text-xs font-medium transition-all">
                Approve All
              </button>
              <button type="button" className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded text-xs font-medium transition-all">
                Reject
              </button>
            </div>
          </div>

          {/* Visual Side-by-Side Diff Panel Simulation */}
          <div className="flex-1 overflow-auto p-4 font-mono text-xs space-y-2">
            <div className="text-xs font-sans text-slate-500 pb-1">📄 src/config/server.ts</div>
            <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900/30">
              <div className="grid grid-cols-2 bg-slate-900/70 p-2 text-[10px] text-slate-500 border-b border-slate-800">
                <div>Original Code</div>
                <div>Agent Revision</div>
              </div>
              <div className="p-3 space-y-1">
                <div className="bg-rose-950/30 text-rose-400 px-1 rounded">- const PORT = process.env.PORT || 3000;</div>
                <div className="bg-emerald-950/30 text-emerald-400 px-1 rounded">+ const PORT = parseInt(process.env.PORT || "8080", 10);</div>
                <div className="text-slate-600">  const app = express();</div>
              </div>
            </div>
          </div>

          {/* 4. BOTTOM DOCK: SERVER LOGS & LIVE WEB VIEW */}
          <div className="h-48 border-t border-slate-800 flex flex-col bg-slate-900/20">
            <div className="h-8 border-b border-slate-800 flex items-center px-4 bg-slate-900/40 justify-between">
              <span className="text-xs font-medium text-slate-400">Live Browser Preview / Hot Reload</span>
              <span className="text-[10px] text-slate-500 font-mono">localhost:8080</span>
            </div>
            <div className="flex-1 p-3 bg-slate-950 flex items-center justify-center text-slate-600 text-xs">
              <div className="text-center">
                <div className="text-slate-500 font-medium">No active dev-server stream detected.</div>
                <div className="text-[11px] mt-1 text-slate-600">Run a startup command via the agent prompt to view hot reloads here.</div>
              </div>
            </div>
          </div>

        </section>

      </main>
    </div>
  );
}