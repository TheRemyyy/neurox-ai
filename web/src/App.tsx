import { Suspense, lazy, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation, Link } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, ChevronLeft, ChevronRight, Menu } from 'lucide-react';
import { Analytics } from '@vercel/analytics/react';
import Home from './pages/Home';

// Lazy load the heavy components
const MarkdownViewer = lazy(() => import('./components/MarkdownViewer').then(module => ({ default: module.MarkdownViewer })));

const allPages = [
  { label: 'Introduction', path: '/' },
  { label: 'Changelog', path: '/docs/CHANGELOG' },
  { label: 'CLI Reference', path: '/docs/cli_reference' },
  { label: 'Architecture', path: '/docs/architecture/system_overview' },
  { label: 'Cortical Hierarchy', path: '/docs/modules/cortical_hierarchy' },
  { label: 'Subcortical Systems', path: '/docs/modules/subcortical_systems' },
  { label: 'Spatial Navigation', path: '/docs/modules/spatial_navigation' },
  { label: 'Neuron Models', path: '/docs/modules/neuron_models' },
  { label: 'Cognitive Upgrades', path: '/docs/modules/cognitive_upgrades' },
  { label: 'Reasoning & Motivation', path: '/docs/modules/reasoning_motivation' },
  { label: 'Problem Solving', path: '/docs/modules/problem_solving' },
  { label: 'MNIST Benchmark', path: '/docs/modules/mnist_benchmark' },
  { label: 'Plasticity', path: '/docs/internals/plasticity' },
  { label: 'Neuromodulation', path: '/docs/internals/neuromodulation' },
  { label: 'GPU Acceleration', path: '/docs/internals/gpu_acceleration' },
  { label: 'Spiking CNN', path: '/docs/modules/spiking_cnn' },
  { label: 'Language & Cognition', path: '/docs/internals/language_cognition' },
  { label: 'Data & Training', path: '/docs/internals/data_training' },
];

const LoadingScreen = () => (
  <div className="flex items-center justify-center h-full w-full py-20">
    <div className="w-6 h-6 border-2 border-accent/30 border-t-accent rounded-full animate-spin"></div>
  </div>
);

const PageWrapper = ({ children }: { children: React.ReactNode }) => (
  <motion.div
    initial={{ opacity: 0, y: 5 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -5 }}
    transition={{ duration: 0.2, ease: "easeOut" }}
    className="max-w-4xl mx-auto py-12 px-6 md:px-12"
  >
    {children}
  </motion.div>
);

const DocsNavigation = () => {
  const location = useLocation();
  const currentPath = location.pathname;
  const normalizedPath = currentPath === '/' ? '/' : currentPath;

  const currentIndex = allPages.findIndex(page => page.path === normalizedPath);
  const prevPage = currentIndex > 0 ? allPages[currentIndex - 1] : null;
  const nextPage = currentIndex < allPages.length - 1 ? allPages[currentIndex + 1] : null;

  if (currentIndex === -1) return null;

  return (
    <div className="mt-16 pt-8 border-t border-border flex flex-col sm:flex-row items-stretch justify-between gap-4">
      {prevPage ? (
        <Link
          to={prevPage.path}
          className="flex-1 flex flex-col gap-2 p-4 rounded-lg border border-border hover:border-accent group transition-all"
        >
          <span className="text-xs text-text-secondary flex items-center gap-1">
            <ChevronLeft size={14} /> Previous
          </span>
          <span className="text-sm font-medium group-hover:text-accent tracking-tight">{prevPage.label}</span>
        </Link>
      ) : <div className="flex-1 hidden sm:block" />}

      {nextPage ? (
        <Link
          to={nextPage.path}
          className="flex-1 flex flex-col items-end gap-2 p-4 rounded-lg border border-border hover:border-accent group transition-all text-right"
        >
          <span className="text-xs text-text-secondary flex items-center gap-1">
            Next <ChevronRight size={14} />
          </span>
          <span className="text-sm font-medium group-hover:text-accent tracking-tight">{nextPage.label}</span>
        </Link>
      ) : <div className="flex-1 hidden sm:block" />}
    </div>
  );
};

const DocsPage = () => {
  const location = useLocation();
  const routePath = location.pathname === '/' ? '/' : location.pathname;
  const fetchPath = routePath === '/' ? '/docs/overview.md' : `${routePath}.md`;
  return (
    <PageWrapper>
      <Suspense fallback={<LoadingScreen />}>
        <MarkdownViewer path={fetchPath} />
      </Suspense>
      <DocsNavigation />
    </PageWrapper>
  );
};

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <Router>
      <div className="flex h-screen min-h-0 bg-background text-text-primary selection:bg-accent/30 font-sans antialiased text-[15px]">
        {/* Mobile Header */}
        <div className="lg:hidden fixed top-0 left-0 right-0 h-14 border-b border-border bg-background/80 backdrop-blur-md z-30 flex items-center px-4 justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-accent rounded flex items-center justify-center text-background">
              <Brain size={14} strokeWidth={2.5} />
            </div>
            <span className="font-semibold tracking-tight">NeuroxAI</span>
          </div>
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 hover:bg-surface rounded-md text-text-secondary"
          >
            <Menu size={20} />
          </button>
        </div>

        <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

        <main className="flex-1 min-h-0 overflow-y-auto pt-14 lg:pt-0">
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<PageWrapper><Home /></PageWrapper>} />
              <Route path="/docs/*" element={<DocsPage />} />
            </Routes>
          </AnimatePresence>
        </main>
        <Analytics />
      </div>
    </Router>
  );
}

export default App;
