import { Routes } from '@angular/router';
import { Component } from '@angular/core';

@Component({ standalone: true, template: '' })
class RouteAnchor {}

export const routes: Routes = [
  { path: '', component: RouteAnchor },
  { path: 'documents', component: RouteAnchor },
  { path: 'chat', component: RouteAnchor },
  { path: 'analytics', component: RouteAnchor },
  { path: 'settings', component: RouteAnchor },
  { path: '**', redirectTo: '' }
];
