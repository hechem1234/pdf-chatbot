import { Component, signal, inject, PLATFORM_ID, ElementRef, ViewChild } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { HttpClient, HttpEventType } from '@angular/common/http';
import { Router, NavigationEnd, RouterLink } from '@angular/router';
import { filter } from 'rxjs';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

type Doc = { name: string; size: number; chunks: number; date: Date; status: string; kind?: 'pdf' | 'image'; id?: string; progress?: number; visionStatus?: string };
type Message = { role: 'user' | 'assistant'; text: string; sources?: { file?: string; source?: string; page?: number; type?: string; document_id?: string; extraction_method?: string }[] };
// Keep this in sync with the backend's `uvicorn main:app --reload --port 8000` command.
const API_BASE = 'http://localhost:8000/api';

@Component({ selector: 'app-root', imports: [RouterLink, FormsModule, CommonModule], templateUrl: './app.html', styleUrl: './app.scss' })
export class App {
  private http = inject(HttpClient); private router = inject(Router);
  private platformId = inject(PLATFORM_ID);
  @ViewChild('messageArea') private messageArea?: ElementRef<HTMLDivElement>;
  page = signal('dashboard');
  docs = signal<Doc[]>([]);
  messages = signal<Message[]>([]);
  question = ''; search = ''; busy = signal(false); dragging = signal(false); notice = signal(''); online = signal(false);
  readonly nav = [{ id: 'dashboard', label: 'Overview', icon: '◫' }, { id: 'documents', label: 'Library', icon: '▤' }, { id: 'chat', label: 'Research studio', icon: '✳' }, { id: 'analytics', label: 'Insights', icon: '⌁' }, { id: 'settings', label: 'System', icon: '⚙' }];
  constructor() {
    this.setPage(this.router.url);
    if (isPlatformBrowser(this.platformId)) this.loadImages();
    this.router.events.pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd)).subscribe(e => this.setPage(e.urlAfterRedirects));
    if (isPlatformBrowser(this.platformId)) this.http.get<{status:string}>(`${API_BASE}/health`).subscribe({ next: () => this.online.set(true), error: () => this.online.set(false) });
  }
  private setPage(url: string) { this.page.set(url.split('/')[1] || 'dashboard'); }
  go(id: string) { this.router.navigateByUrl(id === 'dashboard' ? '/' : `/${id}`); }
  upload(files: FileList | File[] | null) {
    if (!files?.length) return;
    for (const file of Array.from(files)) {
      if (/\.(jpe?g|png|webp)$/i.test(file.name)) { this.uploadImage(file); continue; }
      if (!file.name.toLowerCase().endsWith('.pdf')) { this.flash('Only PDF files can be added.'); continue; }
      const form = new FormData(); form.append('file', file);
      this.docs.update(d => [{ name: file.name, size: file.size, chunks: 0, date: new Date(), status: 'Indexing' }, ...d]);
      this.http.post<{filename:string;chunks:number;status:string}>(`${API_BASE}/documents/upload`, form).subscribe({
        next: res => { this.docs.update(d => d.map(x => x.name === file.name && x.status === 'Indexing' ? {...x, chunks:res.chunks,status:'Ready'} : x)); this.flash(`${file.name} is ready to search.`); },
        error: err => { this.docs.update(d => d.map(x => x.name === file.name && x.status === 'Indexing' ? {...x,status:'Failed'} : x)); this.flash(err.error?.detail || 'Upload failed. Check that the API is running.'); }
      });
    }
  }
  private uploadImage(file: File) {
    if (!['image/jpeg','image/png','image/webp'].includes(file.type)) { this.flash('Choose a JPG, PNG, or WEBP image.'); return; }
    if (file.size > 10 * 1024 * 1024) { this.flash('Images must be 10 MB or smaller.'); return; }
    const form = new FormData(); form.append('file', file);
    const localId = file.name + Date.now();
    this.docs.update(d => [{id:localId,name:file.name,size:file.size,chunks:0,date:new Date(),status:'OCR + Vision',kind:'image',progress:0},...d]);
    this.http.post<any>(API_BASE + '/images/upload',form,{observe:'events',reportProgress:true}).subscribe({
      next:event=>{
        if(event.type===HttpEventType.UploadProgress)this.docs.update(d=>d.map(x=>x.id===localId?{...x,progress:event.total?Math.round(event.loaded*100/event.total):0}:x));
        if(event.type===HttpEventType.Response){const r=event.body;this.docs.update(d=>d.map(x=>x.id===localId?{...x,id:r.image_id,chunks:r.chunks,status:r.vision_status==='ready'?'Ready':'OCR ready · Vision unavailable',visionStatus:r.vision_status,progress:100}:x));this.flash(file.name+' is ready to search.');}
      },
      error:e=>{this.docs.update(d=>d.map(x=>x.id===localId?{...x,status:'Failed'}:x));this.flash(e.error?.detail||'Image processing failed.');}
    });
  }
  loadImages() {
    this.http.get<{images:any[]}>(API_BASE+'/images').subscribe({next:r=>{
      const known=new Set(this.docs().map(x=>x.id));
      const rows=(r.images||[]).filter(x=>!known.has(x.image_id)).map(x=>({id:x.image_id,name:x.filename,size:x.size,chunks:x.chunks,date:new Date(x.created_at),status:x.vision_status==='ready'?'Ready':'OCR ready · Vision unavailable',kind:'image' as const,visionStatus:x.vision_status,progress:100}));
      if(rows.length)this.docs.update(d=>[...rows,...d]);
    }});
  }
  ask() {
    const text = this.question.trim(); if (!text || this.busy()) return;
    this.messages.update(m => [...m, {role:'user',text}]); this.question=''; this.busy.set(true); this.scrollToLatest();
    this.http.post<{answer:string;sources:{file?:string;source?:string;page?:number}[]}>(`${API_BASE}/chat`,{question:text}).subscribe({
      next: res => { this.messages.update(m => [...m,{role:'assistant',text:res.answer,sources:res.sources || []}]); this.busy.set(false); this.scrollToLatest(); },
      error: err => { this.messages.update(m => [...m,{role:'assistant',text:err.error?.detail || 'I could not reach the research service. Check the API connection and try again.'}]); this.busy.set(false); this.scrollToLatest(); }
    });
  }
  resizeComposer(event: Event) {
    const textarea = event.target as HTMLTextAreaElement;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
  }
  private scrollToLatest() {
    setTimeout(() => {
      const area = this.messageArea?.nativeElement;
      if (area) area.scrollTo({ top: area.scrollHeight, behavior: 'smooth' });
    });
  }
  freshChat() { this.messages.set([]); }
  remove(doc: Doc) { this.docs.update(d => d.filter(x => x !== doc)); this.flash('Removed from this session. The API does not currently expose document deletion.'); }
  flash(text:string) { this.notice.set(text); setTimeout(()=>this.notice.set(''),4000); }
  formatSize(n:number) { return n < 1024*1024 ? `${(n/1024).toFixed(0)} KB` : `${(n/1024/1024).toFixed(1)} MB`; }
  get filteredDocs() { return this.docs().filter(d => d.name.toLowerCase().includes(this.search.toLowerCase())); }
}
