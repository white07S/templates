import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

const MermaidDiagram = ({ chart }) => {
  const chartRef = useRef(null);
  const [svg, setSvg] = useState('');
  const [isZoomed, setIsZoomed] = useState(false);

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isZoomed) {
        setIsZoomed(false);
      }
    };

    if (isZoomed) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [isZoomed]);

  useEffect(() => {
    // Configure mermaid with UBS-inspired theme
    mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      themeVariables: {
        primaryColor: '#ffffff',
        primaryTextColor: '#333333',
        primaryBorderColor: '#e60000',
        lineColor: '#666666',
        secondaryColor: '#f5f5f5',
        tertiaryColor: '#e8e8e8',
        background: '#ffffff',
        mainBkg: '#ffffff',
        secondBkg: '#f5f5f5',
        tertiaryBkg: '#e8e8e8',
        secondaryBorderColor: '#666666',
        tertiaryBorderColor: '#d0d0d0',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        fontSize: '14px',
        darkMode: false,
        nodeBorder: '#e60000',
        clusterBkg: '#f5f5f5',
        clusterBorder: '#666666',
        defaultLinkColor: '#666666',
        titleColor: '#333333',
        edgeLabelBackground: '#ffffff',
        actorBorder: '#e60000',
        actorBkg: '#ffffff',
        actorTextColor: '#333333',
        actorLineColor: '#666666',
        signalColor: '#333333',
        signalTextColor: '#333333',
        labelBoxBkgColor: '#f5f5f5',
        labelBoxBorderColor: '#e60000',
        labelTextColor: '#333333',
        loopTextColor: '#333333',
        noteBorderColor: '#e60000',
        noteBkgColor: '#f5f5f5',
        noteTextColor: '#333333',
        activationBorderColor: '#666666',
        activationBkgColor: '#e8e8e8',
        sequenceNumberColor: '#ffffff',
        sectionBkgColor: '#f5f5f5',
        altSectionBkgColor: '#e8e8e8',
        sectionBkgColor2: '#e8e8e8',
        excludeBkgColor: '#f5f5f5',
        taskBorderColor: '#e60000',
        taskBkgColor: '#ffffff',
        taskTextColor: '#333333',
        taskTextDarkColor: '#333333',
        taskTextOutsideColor: '#333333',
        taskTextClickableColor: '#e60000',
        activeTaskBorderColor: '#e60000',
        activeTaskBkgColor: '#f5f5f5',
        gridColor: '#d0d0d0',
        doneTaskBkgColor: '#e8e8e8',
        doneTaskBorderColor: '#666666',
        critBorderColor: '#e60000',
        critBkgColor: '#ffffff',
        todayLineColor: '#e60000',
        personBorder: '#e60000',
        personBkg: '#ffffff',
        labelColor: '#333333',
        errorBkgColor: '#f5f5f5',
        errorTextColor: '#e60000'
      },
      flowchart: {
        htmlLabels: true,
        curve: 'linear',
        rankSpacing: 80,
        nodeSpacing: 80,
        padding: 20,
        useMaxWidth: false,
        defaultRenderer: 'dagre'
      },
      sequence: {
        diagramMarginX: 80,
        diagramMarginY: 50,
        actorMargin: 80,
        width: 200,
        height: 80,
        boxMargin: 15,
        boxTextMargin: 8,
        noteMargin: 15,
        messageMargin: 45,
        mirrorActors: true,
        bottomMarginAdj: 1,
        useMaxWidth: false
      },
      gantt: {
        numberSectionStyles: 4,
        axisFormat: '%Y-%m-%d',
        topAxis: false,
        useMaxWidth: false,
        fontSize: 16
      },
      fontSize: '16px'
    });

    const renderDiagram = async () => {
      if (chartRef.current && chart) {
        try {
          const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
          const { svg } = await mermaid.render(id, chart);
          setSvg(svg);
        } catch (error) {
          console.error('Error rendering Mermaid diagram:', error);
          setSvg('<div class="mermaid-error">Error rendering diagram</div>');
        }
      }
    };

    renderDiagram();
  }, [chart]);

  const handleZoom = () => {
    setIsZoomed(!isZoomed);
  };

  return (
    <div className="relative my-8">
      <div
        ref={chartRef}
        className={`
          bg-gray-50 p-8 transition-all duration-300 cursor-pointer
          ${isZoomed
            ? 'fixed inset-4 z-50 bg-white shadow-2xl overflow-auto'
            : 'w-full overflow-x-auto overflow-y-visible'
          }
        `}
        onClick={handleZoom}
        style={{
          minHeight: isZoomed ? '400px' : 'auto',
          maxWidth: isZoomed ? 'none' : '100%'
        }}
      >
        {svg && (
          <div
            dangerouslySetInnerHTML={{ __html: svg }}
            className={`
              ${isZoomed
                ? 'flex justify-center items-center min-h-full'
                : 'min-w-max'
              }
            `}
            style={{
              transform: isZoomed ? 'scale(1.2)' : 'scale(1)',
              transformOrigin: 'center'
            }}
          />
        )}
      </div>
      {!isZoomed && (
        <div className="absolute top-4 right-4 bg-red-600 text-white px-3 py-1 text-sm pointer-events-none">
          Click to zoom
        </div>
      )}
      {isZoomed && (
        <div className="fixed top-4 right-4 z-50 bg-red-600 text-white px-4 py-2 text-sm">
          Press ESC or click to close
        </div>
      )}
    </div>
  );
};

export default MermaidDiagram;