import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

const GridBackground = () => {
    const [columns, setColumns] = useState(0);
    const [rows, setRows] = useState(0);
    const containerRef = useRef(null);

    useEffect(() => {
        const calculateGrid = () => {
            if (containerRef.current) {
                const width = containerRef.current.clientWidth;
                const height = containerRef.current.clientHeight;
                const size = 50; // Tile size in pixels
                setColumns(Math.ceil(width / size));
                setRows(Math.ceil(height / size));
            }
        };

        calculateGrid();
        window.addEventListener('resize', calculateGrid);
        return () => window.removeEventListener('resize', calculateGrid);
    }, []);

    const handleHover = (e) => {
        const tile = e.target;
        // Set opacity to 1 immediately
        tile.style.backgroundColor = 'rgba(6, 182, 212, 0.2)';
        tile.style.borderColor = 'rgba(6, 182, 212, 0.5)';
        tile.style.transition = 'none'; // Instant on

        // Clear previous timeout if exists
        if (tile.timeoutId) clearTimeout(tile.timeoutId);

        // Start fade out after a small delay
        tile.timeoutId = setTimeout(() => {
            tile.style.transition = 'background-color 1s ease-out, border-color 1s ease-out'; // Slow fade
            tile.style.backgroundColor = '';
            tile.style.borderColor = '';
        }, 50);
    };

    return (
        <div
            ref={containerRef}
            className="fixed inset-0 z-[-1] overflow-hidden pointer-events-none"
        >
            <div
                className="absolute inset-0 grid gap-[1px] pointer-events-auto"
                style={{
                    gridTemplateColumns: `repeat(${columns}, 1fr)`,
                    gridTemplateRows: `repeat(${rows}, 1fr)`,
                }}
            >
                {Array.from({ length: columns * rows }).map((_, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{
                            duration: 0.5,
                            delay: (i % columns) * 0.02 + (Math.floor(i / columns) * 0.02), // Stagger from top-left
                            ease: "easeOut"
                        }}
                        onMouseEnter={handleHover}
                        className={clsx(
                            "relative bg-slate-900/50 border border-white/5",
                            // Removed hover classes as we handle it manually for the trail effect
                        )}
                    >
                        {/* Dot in center of tile */}
                        <div className="absolute top-1/2 left-1/2 w-[2px] h-[2px] bg-slate-800 -translate-x-1/2 -translate-y-1/2" />
                    </motion.div>
                ))}
            </div>

            {/* Global Gradient Overlay to fade edges */}
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#09090b] pointer-events-none" />
            <div className="absolute inset-0 bg-gradient-to-r from-[#09090b] via-transparent to-[#09090b] pointer-events-none" />
        </div>
    );
};

export default GridBackground;
