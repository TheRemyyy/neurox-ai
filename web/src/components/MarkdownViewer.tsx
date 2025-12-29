import { useEffect, useState } from 'react';
import { marked } from 'marked';
import hljs from 'highlight.js';
import 'highlight.js/styles/github-dark.css';

interface MarkdownViewerProps {
  path: string;
}

export const MarkdownViewer = ({ path }: MarkdownViewerProps) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(path)
      .then((res) => res.text())
      .then((text) => {
        // Set options for better gfm support and line breaks
        const html = marked.parse(text, { breaks: true, gfm: true }) as string;
        setContent(html);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load markdown:', err);
        setContent('<p className="text-red-500">Failed to load documentation segment.</p>');
        setLoading(false);
      });
  }, [path]);

  useEffect(() => {
    if (!loading) {
      document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block as HTMLElement);
      });
    }
  }, [content, loading]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-t-accent border-border rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div 
      className="prose prose-invert prose-pre:bg-surface prose-pre:border prose-pre:border-border max-w-none
                 prose-headings:font-semibold prose-headings:tracking-tight
                 prose-a:text-accent prose-a:no-underline hover:prose-a:underline
                 prose-code:text-accent prose-code:bg-surface prose-code:px-1 prose-code:rounded"
      dangerouslySetInnerHTML={{ __html: content }}
    />
  );
};
