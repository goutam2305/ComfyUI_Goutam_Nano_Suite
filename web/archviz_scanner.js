import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Gemini.ArchViz.Scanner",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Gemini_ArchViz_Scanner") {
            return;
        }

        // ─── SUPPRESS STANDARD IMAGE PREVIEW ───
        // ComfyUI draws node.imgs inside onDrawBackground.
        // Override to prevent double-rendering (our widget handles it).
        nodeType.prototype.onDrawBackground = function (ctx) {
            // Empty — custom widget renders the image instead.
        };

        // ─── onExecuted: Load images from backend response ───
        // The Python backend returns { "ui": {"images": [...]}, "result": (...) }
        // We must process this to create Image objects and store them in node.imgs.
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);

            // Process ui.images from the backend
            if (message && message.images && message.images.length > 0) {
                const imgData = message.images[0];
                const img = new Image();
                img.src = api.apiURL(
                    `/view?filename=${encodeURIComponent(imgData.filename)}` +
                    `&subfolder=${encodeURIComponent(imgData.subfolder || "")}` +
                    `&type=${encodeURIComponent(imgData.type || "temp")}`
                );
                img.onload = () => {
                    this.setDirtyCanvas(true, true);
                };
                this.imgs = [img];
                this.setDirtyCanvas(true, true);
            }
        };

        // ─── NODE CREATION ───
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            this.setSize([500, 750]);
            this.defaults_applied = false;

            // Sort: Move ARCHVIZ_PREVIEW to bottom of widget list
            const self = this;
            setTimeout(() => {
                if (self.widgets) {
                    const pw = self.widgets.find(x => x.name === "ARCHVIZ_PREVIEW");
                    if (pw && self.widgets[self.widgets.length - 1] !== pw) {
                        const idx = self.widgets.indexOf(pw);
                        self.widgets.splice(idx, 1);
                        self.widgets.push(pw);
                        self.setDirtyCanvas(true, true);
                    }
                }
            }, 500);

            // ─── CUSTOM WIDGET ───
            const previewWidget = {
                name: "ARCHVIZ_PREVIEW",
                type: "ARCHVIZ_PREVIEW",
                value: null,
                draw: function (ctx, node, w, y, rawH) {
                    if (y < 50) return;

                    // LiteGraph passes h=20 (default) ignoring computeSize.
                    // Compute actual available height from node size.
                    const MARGIN = 10;
                    const h = Math.max(200, node.size[1] - y - MARGIN);

                    // ─── READ node.imgs DIRECTLY ───
                    let img = null;
                    if (node.imgs && node.imgs.length > 0) {
                        img = node.imgs[0];
                    }

                    // Background
                    ctx.fillStyle = "#1a1a1a";
                    ctx.fillRect(0, y, w, h);

                    if (img && img.naturalWidth) {
                        // ─── DEFAULTS (One-Time) ───
                        const wTop = node.widgets.find(x => x.name === "crop_top");
                        const wLeft = node.widgets.find(x => x.name === "crop_left");
                        const wH = node.widgets.find(x => x.name === "crop_bottom");
                        const wW = node.widgets.find(x => x.name === "crop_right");

                        if (!node.defaults_applied && wTop && wLeft && wH && wW) {
                            if (wTop.value === 0 && wLeft.value === 0 && wH.value === 0 && wW.value === 0) {
                                wTop.value = 2048;
                                wLeft.value = 2048;
                                wH.value = 512;
                                wW.value = 512;
                            }
                            node.defaults_applied = true;
                        }

                        // ─── FIT IMAGE TO WIDGET ───
                        const iw = img.naturalWidth;
                        const ih = img.naturalHeight;
                        const ratioImg = iw / ih;
                        const ratioArea = w / h;

                        let dw, dh, dx, dy;
                        if (ratioImg > ratioArea) {
                            dw = w; dh = w / ratioImg;
                            dx = 0; dy = y + (h - dh) * 0.5;
                        } else {
                            dh = h; dw = h * ratioImg;
                            dy = y; dx = (w - dw) * 0.5;
                        }

                        ctx.drawImage(img, dx, dy, dw, dh);
                        this.hitArea = { x: dx, y: dy, w: dw, h: dh, iw, ih };

                        // ─── CROP OVERLAY ───
                        let top = wTop ? wTop.value : 0;
                        let left = wLeft ? wLeft.value : 0;
                        let cropH = wH ? wH.value : 100;
                        let cropW = wW ? wW.value : 100;

                        const sX = dw / iw;
                        const sY = dh / ih;
                        const rx = dx + left * sX;
                        const ry = dy + top * sY;
                        const rw = cropW * sX;
                        const rh = cropH * sY;

                        // Dim outside crop (save/restore prevents clip leak)
                        ctx.save();
                        ctx.fillStyle = "rgba(0,0,0,0.5)";
                        ctx.beginPath();
                        ctx.rect(dx, dy, dw, dh);
                        ctx.rect(rx, ry, rw, rh);
                        ctx.clip("evenodd");
                        ctx.fill();
                        ctx.restore();

                        // Crop border
                        ctx.strokeStyle = "#0F0";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(rx, ry, rw, rh);

                        // Dimensions label
                        ctx.fillStyle = "#0F0";
                        ctx.font = "bold 12px monospace";
                        ctx.textAlign = "left";
                        ctx.fillText(`${cropW}x${cropH}`, rx + 5, ry + 15);

                    } else {
                        ctx.fillStyle = "#555";
                        ctx.textAlign = "center";
                        ctx.font = "italic 14px Arial";
                        ctx.fillText("Waiting for generation...", w / 2, y + h / 2);
                    }
                },
                computeSize: function (width) {
                    return [width, 450];
                },
                // ─── MOUSE INTERACTION ───
                // LiteGraph routes widget-area clicks to widget.mouse(),
                // bypassing node.onMouseDown entirely.
                mouse: function (event, pos, node) {
                    const w = this;
                    if (!w.hitArea) return false;

                    // ─── ASPECT RATIO HELPER ───
                    const RATIO_MAP = {
                        "Free": null, "1:1": 1, "4:3": 4 / 3, "3:2": 3 / 2,
                        "16:9": 16 / 9, "21:9": 21 / 9, "9:16": 9 / 16,
                        "2:3": 2 / 3, "3:4": 3 / 4
                    };
                    function getLockedRatio() {
                        const presetW = node.widgets.find(x => x.name === "aspect_preset");
                        if (!presetW) return null;
                        const preset = presetW.value;
                        if (preset === "Custom") {
                            const customW = node.widgets.find(x => x.name === "custom_ratio");
                            return customW ? customW.value : null;
                        }
                        return RATIO_MAP[preset] || null;
                    }

                    const area = w.hitArea;
                    const mx = pos[0], my = pos[1];

                    // Only handle events inside the image area
                    if (mx < area.x || mx > area.x + area.w ||
                        my < area.y || my > area.y + area.h) return false;

                    const sX = area.iw / area.w;
                    const sY = area.ih / area.h;
                    const imgX = (mx - area.x) * sX;
                    const imgY = (my - area.y) * sY;

                    if (event.type === "pointerdown") {
                        const wTop = node.widgets.find(x => x.name === "crop_top");
                        const wLeft = node.widgets.find(x => x.name === "crop_left");
                        const wH = node.widgets.find(x => x.name === "crop_bottom");
                        const wW = node.widgets.find(x => x.name === "crop_right");
                        if (!wTop || !wLeft || !wH || !wW) return false;

                        // Current crop box in image space
                        const left = wLeft.value || 0;
                        const top = wTop.value || 0;
                        const width = wW.value || 0;
                        const height = wH.value || 0;
                        const right = left + width;
                        const bottom = top + height;

                        // Hit detection threshold (15 pixels in image space)
                        const thresh = 15;
                        let mode = 'new';
                        let handle = null;

                        // Check corners
                        if (Math.abs(imgX - left) < thresh && Math.abs(imgY - top) < thresh) {
                            mode = 'resize'; handle = 'tl';
                        } else if (Math.abs(imgX - right) < thresh && Math.abs(imgY - top) < thresh) {
                            mode = 'resize'; handle = 'tr';
                        } else if (Math.abs(imgX - left) < thresh && Math.abs(imgY - bottom) < thresh) {
                            mode = 'resize'; handle = 'bl';
                        } else if (Math.abs(imgX - right) < thresh && Math.abs(imgY - bottom) < thresh) {
                            mode = 'resize'; handle = 'br';
                        }
                        // Check edges
                        else if (Math.abs(imgX - left) < thresh && imgY > top && imgY < bottom) {
                            mode = 'resize'; handle = 'l';
                        } else if (Math.abs(imgX - right) < thresh && imgY > top && imgY < bottom) {
                            mode = 'resize'; handle = 'r';
                        } else if (Math.abs(imgY - top) < thresh && imgX > left && imgX < right) {
                            mode = 'resize'; handle = 't';
                        } else if (Math.abs(imgY - bottom) < thresh && imgX > left && imgX < right) {
                            mode = 'resize'; handle = 'b';
                        }
                        // Check interior (move)
                        else if (imgX > left && imgX < right && imgY > top && imgY < bottom) {
                            mode = 'move';
                        }

                        node.dragState = {
                            mode, handle,
                            startX: imgX, startY: imgY,
                            origLeft: left, origTop: top,
                            origW: width, origH: height,
                            wTop, wLeft, wH, wW, area
                        };

                        if (mode === 'new') {
                            wLeft.value = Math.round(imgX);
                            wTop.value = Math.round(imgY);
                            wW.value = 0;
                            wH.value = 0;
                        }

                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    if (event.type === "pointermove" && node.dragState) {
                        const d = node.dragState;
                        const a = d.area;
                        const scX = a.iw / a.w;
                        const scY = a.ih / a.h;
                        const ratio = getLockedRatio();

                        let currX = Math.max(0, Math.min((mx - a.x) * scX, a.iw));
                        let currY = Math.max(0, Math.min((my - a.y) * scY, a.ih));

                        const dx = currX - d.startX;
                        const dy = currY - d.startY;

                        let newLeft, newTop, newW, newH;

                        if (d.mode === 'new') {
                            newLeft = Math.min(currX, d.startX);
                            newTop = Math.min(currY, d.startY);
                            newW = Math.abs(currX - d.startX);
                            newH = Math.abs(currY - d.startY);

                            // Enforce ratio: width drives height
                            if (ratio && newW > 0) {
                                newH = newW / ratio;
                                // If dragging upward, adjust top
                                if (currY < d.startY) {
                                    newTop = d.startY - newH;
                                }
                            }
                        } else if (d.mode === 'move') {
                            newLeft = Math.max(0, Math.min(a.iw - d.origW, d.origLeft + dx));
                            newTop = Math.max(0, Math.min(a.ih - d.origH, d.origTop + dy));
                            newW = d.origW;
                            newH = d.origH;
                        } else if (d.mode === 'resize') {
                            let left = d.origLeft;
                            let top = d.origTop;
                            let right = d.origLeft + d.origW;
                            let bottom = d.origTop + d.origH;

                            if (d.handle.includes('l')) left = Math.min(right - 1, Math.max(0, d.origLeft + dx));
                            if (d.handle.includes('r')) right = Math.max(left + 1, Math.min(a.iw, d.origLeft + d.origW + dx));
                            if (d.handle.includes('t')) top = Math.min(bottom - 1, Math.max(0, d.origTop + dy));
                            if (d.handle.includes('b')) bottom = Math.max(top + 1, Math.min(a.ih, d.origTop + d.origH + dy));

                            newW = right - left;
                            newH = bottom - top;

                            // Enforce ratio during resize
                            if (ratio && newW > 0 && newH > 0) {
                                const h = d.handle;
                                // Horizontal handles: width is primary, adjust height
                                if (h === 'l' || h === 'r') {
                                    newH = newW / ratio;
                                    // Center vertically relative to original box
                                    top = d.origTop + (d.origH - newH) / 2;
                                    bottom = top + newH;
                                }
                                // Vertical handles: height is primary, adjust width
                                else if (h === 't' || h === 'b') {
                                    newW = newH * ratio;
                                    left = d.origLeft + (d.origW - newW) / 2;
                                    right = left + newW;
                                }
                                // Corner handles: width drives height
                                else {
                                    newH = newW / ratio;
                                    if (h.includes('t')) {
                                        top = bottom - newH;
                                    } else {
                                        bottom = top + newH;
                                    }
                                }
                            }

                            newLeft = left;
                            newTop = top;
                            newW = right - left;
                            newH = bottom - top;
                        }

                        // Clamp to image bounds
                        newLeft = Math.max(0, newLeft);
                        newTop = Math.max(0, newTop);
                        if (newLeft + newW > a.iw) newW = a.iw - newLeft;
                        if (newTop + newH > a.ih) newH = a.ih - newTop;

                        d.wLeft.value = Math.round(newLeft);
                        d.wTop.value = Math.round(newTop);
                        d.wW.value = Math.round(Math.max(1, newW));
                        d.wH.value = Math.round(Math.max(1, newH));

                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    if (event.type === "pointerup" && node.dragState) {
                        node.dragState = null;
                        return true;
                    }

                    return false;
                }
            };

            this.addCustomWidget(previewWidget);
            this.dragState = null;
        };
    }
});
