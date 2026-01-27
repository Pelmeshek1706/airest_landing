const PARTICLE_COUNT = 30;
const PARTICLE_SPEED = 2;
const PARTICLE_LIFESPAN = 60;
const SHOCKWAVE_SPEED = 4;
const SHOCKWAVE_MAX_RADIUS = 100;

class Particle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * PARTICLE_SPEED + 1;
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed;
        this.life = PARTICLE_LIFESPAN;
        this.initialLife = this.life;
        this.radius = Math.random() * 2 + 1;
    }

    update() {
        this.life--;
        this.x += this.vx;
        this.y += this.vy;
    }

    draw(ctx) {
        const opacity = this.life / this.initialLife;
        ctx.fillStyle = `rgba(14, 165, 233, ${opacity})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
    }
}


const Animator = {
    ctx: null,
    canvas: null,
    animationFrameId: null,

    pulseFactor: 0,
    particles: [],
    shockwave: { active: false, x: 0, y: 0, radius: 0, opacity: 1 },
    
    idleOpacity: 1.0,
    collectingOpacity: 0.0,
    targetIdleOpacity: 1.0,
    targetCollectingOpacity: 0.0,

    init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    },

    setPhase(phase) {
        if (phase === 'collecting') {
            this.targetIdleOpacity = 0.0;
            this.targetCollectingOpacity = 1.0;
        } else if (phase === 'awaiting_trigger') {
            this.targetIdleOpacity = 1.0;
            this.targetCollectingOpacity = 0.0;
        } else {
            this.targetIdleOpacity = 0.0;
            this.targetCollectingOpacity = 0.0;
        }
    },

    draw(status, fps) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.updateEffects();
        this.drawEffects();
        
        this.drawFPS(fps);
        
        if (!status || !status.display_info) return;
        
        const info = status.display_info;

        switch (info.type) {
            case 'instruction_text':
            case 'message':
                this.drawInstruction(info.text);
                break;
            case 'fixation_dot':
            case 'test_dot':
                this.drawCalibrationPoint(status);
                break;
        }
    },

    drawInstruction(text) {
        this.ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
        this.ctx.font = '600 28px "Space Grotesk", sans-serif';
        const lines = text.split('\n');
        lines.forEach((line, i) => {
            const m = this.ctx.measureText(line);
            this.ctx.fillText(line, (this.canvas.width - m.width) / 2, 150 + i * 50);
        });
    },

    drawCalibrationPoint(status) {
        const info = status.display_info;
        if (!info.target_point) return;

        this.drawAwaiting(info, this.idleOpacity);
        this.drawCollecting(info, this.collectingOpacity);
        this.drawInnerPoint(info);

        if (status.phase === 'point_completed') {
            this.drawCompleted(info);
        }
        
        if (info.type === 'test_dot' && info.estimated_gaze) {
            this.drawGaze(info.estimated_gaze);
        }
    },
    
    drawAwaiting(info, opacity) {
        if (opacity <= 0) return;
        const [targetX, targetY] = info.target_point;

        const pulseProgress = (Math.sin(this.pulseFactor * 2) + 1) / 2;
        const pulseRadius = info.inner_circle_radius + 10 + (pulseProgress * 15);
        
        this.ctx.strokeStyle = `rgba(99, 102, 241, ${opacity * 0.7})`;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(targetX, targetY, pulseRadius, 0, Math.PI * 2);
        this.ctx.stroke();
    },

    drawCollecting(info, opacity) {
        if (opacity <= 0) return;
        const [targetX, targetY] = info.target_point;
        
        this.ctx.strokeStyle = `rgba(14, 165, 233, ${opacity * 0.9})`;
        this.ctx.lineWidth = 4;
        this.ctx.beginPath();
        this.ctx.arc(targetX, targetY, info.outer_circle_radius, 0, Math.PI * 2);
        this.ctx.stroke();
    },
    
    drawInnerPoint(info) {
        const [targetX, targetY] = info.target_point;
        this.ctx.fillStyle = '#2563eb';
        this.ctx.beginPath();
        this.ctx.arc(targetX, targetY, info.inner_circle_radius, 0, Math.PI * 2);
        this.ctx.fill();
    },

    drawCompleted(info) {
        const [targetX, targetY] = info.target_point;
        this.ctx.fillStyle = '#10b981';
        this.ctx.beginPath();
        this.ctx.arc(targetX, targetY, info.inner_circle_radius, 0, Math.PI * 2);
        this.ctx.fill();
    },
    
    drawGaze(gaze) {
        const [gx, gy] = gaze;
        this.ctx.strokeStyle = '#f59e0b';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(gx - 10, gy);
        this.ctx.lineTo(gx + 10, gy);
        this.ctx.moveTo(gx, gy - 10);
        this.ctx.lineTo(gx, gy + 10);
        this.ctx.stroke();
    },
    
    drawFPS(fps) {
        if (fps > 0) {
            this.ctx.fillStyle = 'rgba(15, 23, 42, 0.6)';
            this.ctx.font = '12px "Space Grotesk", sans-serif';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(`Processing FPS: ${fps.toFixed(1)}`, 10, 20);
        }
    },

    triggerCompletionEffect(x, y) {
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            this.particles.push(new Particle(x, y));
        }
        this.shockwave = { active: true, x: x, y: y, radius: 0, opacity: 1 };
    },
    
    updateEffects() {
        const lerpFactor = 0.15;
        this.idleOpacity += (this.targetIdleOpacity - this.idleOpacity) * lerpFactor;
        this.collectingOpacity += (this.targetCollectingOpacity - this.collectingOpacity) * lerpFactor;

        this.particles.forEach(p => p.update());
        this.particles = this.particles.filter(p => p.life > 0);

        if (this.shockwave.active) {
            this.shockwave.radius += SHOCKWAVE_SPEED;
            this.shockwave.opacity = 1 - (this.shockwave.radius / SHOCKWAVE_MAX_RADIUS);
            if (this.shockwave.opacity <= 0) {
                this.shockwave.active = false;
            }
        }
    },

    drawEffects() {
        this.particles.forEach(p => p.draw(this.ctx));

        if (this.shockwave.active) {
            this.ctx.strokeStyle = `rgba(14, 165, 233, ${this.shockwave.opacity})`;
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.arc(this.shockwave.x, this.shockwave.y, this.shockwave.radius, 0, Math.PI * 2);
            this.ctx.stroke();
        }
    },
    
    run(callback) {
        const loop = (timestamp) => {
            this.pulseFactor = timestamp * 0.002;
            callback();
            this.animationFrameId = requestAnimationFrame(loop);
        };
        this.animationFrameId = requestAnimationFrame(loop);
    },

    stop() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }
};
