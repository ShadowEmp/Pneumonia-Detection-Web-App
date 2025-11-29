import React, { useEffect, useRef } from 'react';

const NeuralField = () => {
    const canvasRef = useRef(null);
    const particles = useRef([]);
    const mouse = useRef({ x: -1000, y: -1000 }); // Start off-screen

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let animationFrameId;

        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            initParticles();
        };

        const initParticles = () => {
            particles.current = [];
            const density = 45; // Spacing between particles
            const cols = Math.floor(canvas.width / density);
            const rows = Math.floor(canvas.height / density);

            for (let i = 0; i <= cols; i++) {
                for (let j = 0; j <= rows; j++) {
                    const x = i * density;
                    const y = j * density;

                    // Add some randomness to initial position for organic look
                    const randomX = x + (Math.random() - 0.5) * 20;
                    const randomY = y + (Math.random() - 0.5) * 20;

                    particles.current.push({
                        x: randomX,
                        y: randomY,
                        originX: randomX,
                        originY: randomY,
                        vx: 0,
                        vy: 0,
                        size: Math.random() * 2.5 + 1.0, // Increased size
                        color: `hsl(${Math.random() * 40 + 180}, 100%, 75%)`, // Brighter/Vibrant Cyan
                        phaseX: Math.random() * Math.PI * 2, // Random starting phase for X
                        phaseY: Math.random() * Math.PI * 2  // Random starting phase for Y
                    });
                }
            }
        };

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        const handleMouseMove = (e) => {
            mouse.current = { x: e.clientX, y: e.clientY };
        };

        const handleMouseLeave = () => {
            mouse.current = { x: -1000, y: -1000 };
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseleave', handleMouseLeave);

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const repulsionRadius = 150;
            const repulsionStrength = 2.5;
            const springStrength = 0.05;
            const friction = 0.9;

            // 0. Organic "Breathing" Motion (The "Alive" Vibe)
            const time = Date.now() * 0.001;
            const floatRadius = 15; // How far they wander
            const floatSpeed = 0.5;

            particles.current.forEach((p) => {
                // Calculate moving target position
                const targetX = p.originX + Math.cos(time * floatSpeed + p.phaseX) * floatRadius;
                const targetY = p.originY + Math.sin(time * floatSpeed + p.phaseY) * floatRadius;

                // 1. Repulsion from mouse
                const dx = mouse.current.x - p.x;
                const dy = mouse.current.y - p.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < repulsionRadius) {
                    const force = (repulsionRadius - distance) / repulsionRadius;
                    const angle = Math.atan2(dy, dx);
                    const fx = Math.cos(angle) * force * repulsionStrength;
                    const fy = Math.sin(angle) * force * repulsionStrength;

                    p.vx -= fx;
                    p.vy -= fy;
                }

                // 2. Spring back to MOVING target (not static origin)
                const ox = targetX - p.x;
                const oy = targetY - p.y;

                p.vx += ox * springStrength;
                p.vy += oy * springStrength;

                // 3. Physics update
                p.vx *= friction;
                p.vy *= friction;
                p.x += p.vx;
                p.y += p.vy;

                // Draw particle with glow
                ctx.shadowBlur = 15; // Glow strength
                ctx.shadowColor = p.color;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.fill();

                // Reset shadow to avoid affecting other elements heavily
                ctx.shadowBlur = 0;
            });

            // Draw connections (optimized)
            ctx.lineWidth = 0.6; // Thicker lines for better visibility
            const maxDist = 120;
            const maxDistSq = maxDist * maxDist;

            for (let i = 0; i < particles.current.length; i++) {
                const p1 = particles.current[i];

                // Check more neighbors for better connectivity
                for (let j = i + 1; j < Math.min(i + 150, particles.current.length); j++) {
                    const p2 = particles.current[j];
                    const dx = p1.x - p2.x;
                    const dy = p1.y - p2.y;
                    const distSq = dx * dx + dy * dy;

                    if (distSq < maxDistSq) {
                        // Dynamic opacity based on distance (closer = stronger)
                        const opacity = 1 - (distSq / maxDistSq);
                        ctx.strokeStyle = `rgba(6, 182, 212, ${opacity * 0.6})`;

                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseleave', handleMouseLeave);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 pointer-events-none z-[1]" // Background layer
            style={{ mixBlendMode: 'screen' }} // Add nice blend mode
        />
    );
};

export default NeuralField;
