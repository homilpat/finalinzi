---
name: Senior Vitality System
colors:
  surface: '#f8f9ff'
  surface-dim: '#c5dcfb'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eef4ff'
  surface-container: '#e4efff'
  surface-container-high: '#dae9ff'
  surface-container-highest: '#d0e4ff'
  on-surface: '#021d33'
  on-surface-variant: '#414751'
  inverse-surface: '#1a324a'
  inverse-on-surface: '#e9f1ff'
  outline: '#717783'
  outline-variant: '#c1c7d3'
  surface-tint: '#0060ac'
  primary: '#005da7'
  on-primary: '#ffffff'
  primary-container: '#2976c7'
  on-primary-container: '#fdfcff'
  inverse-primary: '#a4c9ff'
  secondary: '#006b5a'
  on-secondary: '#ffffff'
  secondary-container: '#97f0da'
  on-secondary-container: '#00705e'
  tertiary: '#994030'
  on-tertiary: '#ffffff'
  tertiary-container: '#b85846'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d4e3ff'
  primary-fixed-dim: '#a4c9ff'
  on-primary-fixed: '#001c39'
  on-primary-fixed-variant: '#004883'
  secondary-fixed: '#9af3dc'
  secondary-fixed-dim: '#7ed7c1'
  on-secondary-fixed: '#00201a'
  on-secondary-fixed-variant: '#005143'
  tertiary-fixed: '#ffdad3'
  tertiary-fixed-dim: '#ffb4a6'
  on-tertiary-fixed: '#3f0300'
  on-tertiary-fixed-variant: '#7d2c1e'
  background: '#f8f9ff'
  on-background: '#021d33'
  surface-variant: '#d0e4ff'
typography:
  headline-lg:
    fontFamily: Noto Sans KR
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 60px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Noto Sans KR
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 54px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Noto Sans KR
    fontSize: 28px
    fontWeight: '500'
    lineHeight: 48px
  body-md:
    fontFamily: Noto Sans KR
    fontSize: 24px
    fontWeight: '400'
    lineHeight: 40px
  label-md:
    fontFamily: Noto Sans KR
    fontSize: 20px
    fontWeight: '700'
    lineHeight: 28px
    letterSpacing: 0.02em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  touch-target-min: 72px
  button-height-min: 80px
  container-padding: 24px
  stack-gap: 32px
  safe-zone-bottom: 88px
  back-button-tap: 52px
---

## Brand & Style

This design system is engineered specifically for the elderly, prioritizing extreme legibility, cognitive ease, and physical accessibility. The brand personality is "Warmly Authoritative"—it acts as a professional health partner that feels like a caring family member.

The visual style is **Soft-Tactile Modernism**. It avoids the clutter of traditional medical interfaces in favor of a clean, high-contrast environment with large, "squishy" interactive elements that feel safe to touch. We utilize a bright, optimistic palette to counteract the clinical coldness often associated with health apps, evoking a sense of energy and renewed vitality. 

Key principles:
- **Zero Ambiguity:** Every icon has a label; every button looks like a physical object.
- **Cognitive Calm:** Maximum whitespace to prevent information overload.
- **Physical Comfort:** Large tap targets designed for reduced motor precision.

## Colors

The color strategy focuses on high-contrast accessibility (WCAG AAA standards) and emotional resonance.

- **Primary Blue (#4A90E2):** Used for primary actions and "Trust" elements. It provides a stable, professional foundation.
- **Mint Vitality (#7FD8C2):** Used for positive health indicators, success states, and calming UI sections.
- **Warm Coral Accent (#FF8F7A):** A friendly highlight color used sparingly for critical alerts or "Heart" related features to draw immediate attention without causing alarm.
- **Text Navy (#243B53):** Replaces pure black to reduce eye strain while maintaining maximum contrast against the Light Background (#F7FAFC).

## Typography

Typography is the backbone of this system. We use **Noto Sans KR** for its exceptional legibility in the Korean script and its clean, modern aesthetic. 

- **Size & Spacing:** Minimum body size is 24px. Line heights are set to a generous 1.8x - 2.0x to ensure characters do not "blur" together for users with presbyopia.
- **Weight:** We lean toward Medium and Bold weights. Thin fonts are strictly prohibited as they disappear under high-brightness or low-vision conditions.
- **Hierarchy:** Use clear, large headings to signpost where the user is at all times.

## Layout & Spacing

The layout follows a **Fluid Vertical Stack** model. Since the target audience may struggle with complex horizontal layouts or hidden menus, the UI grows vertically with clear, distinct sections.

- **Grid:** A simple 2-column grid is the maximum complexity allowed; a single-column stack is preferred.
- **Tap Targets:** Every interactive element must be at least 72px in height/width to accommodate tremors or reduced dexterity.
- **Margins:** A consistent 24px outer margin ensures content doesn't bleed into the physical edges of the device.
- **Safe Zones:** A mandatory 88px bottom safe zone prevents accidental triggers of OS-level navigation gestures (like the home bar).

## Elevation & Depth

We use **Ambient Shadows** and **Tonal Layering** to create a sense of physical space. This helps the user understand what can be clicked and what is static information.

- **Shadows:** Use a soft, diffused shadow (`0 4px 20px rgba(0,0,0,0.10)`) for cards. This "lifts" the card off the background, signaling it as a container of information.
- **Surface Contrast:** Use pure White (#FFFFFF) for interactive cards against the Light Background (#F7FAFC).
- **Depth:** Active states for buttons should involve a slight "press" effect (reducing shadow and darkening the color slightly) to provide immediate haptic and visual feedback.

## Shapes

The shape language is dominated by **friendly, exaggerated rounds**. 

- **Buttons:** Use 20px corner radius for a "pill-like" but structured feel.
- **Cards:** Use 24px - 32px corner radius to make the interface feel soft and non-threatening.
- **Icons:** Always contained within rounded circles or squares with a minimum radius of 16px to maintain the "contained" and safe aesthetic.

## Components

Components are designed for high visibility and "one-tap" interaction.

- **Primary Buttons:** Minimum 80px height. Always full-width to provide the largest possible hit area. Use Primary Blue with White text.
- **Cards:** Must have 28px internal padding. They should contain a maximum of one primary action and two lines of text to maintain focus.
- **Navigation:** The "Back" button is a persistent anchor at the top-left, with a minimum 52px tap area and a clear "뒤로" (Back) label.
- **Icons:** Large (56px+) and simple. Icons must never stand alone; they must always be accompanied by a 20px+ label below them.
- **Selection (Radio/Checkbox):** Use "Card-style" selection where the entire row is the tap target, turning Primary Blue or Mint when selected.
- **Input Fields:** Extra-thick borders (2px) and large placeholder text. Active fields should have a Coral or Blue glow to indicate focus.
- **Interactive Limit:** To prevent cognitive overload, no screen should feature more than 3 distinct interactive choices (excluding the back button).