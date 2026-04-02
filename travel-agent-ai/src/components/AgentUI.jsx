import React, { useState, useRef, useEffect } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Loader2, MapPin, Play, FileText, CheckCircle2, AlertCircle, ArrowRight, Calendar as CalendarIcon, Download, ChevronLeft, ChevronRight, Clock, Trash2, Brain, User } from 'lucide-react';
import { format, parseISO, startOfMonth, endOfMonth, eachDayOfInterval, startOfWeek, endOfWeek, addMonths, subMonths, isToday, isSameMonth } from 'date-fns';

export default function AgentUI() {
  const [input, setInput] = useState('');
  const [logs, setLogs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [calendarEvents, setCalendarEvents] = useState([]);
  const [activeTab, setActiveTab] = useState('itinerary');
  const [history, setHistory] = useState([]);
  const [activeTrip, setActiveTrip] = useState(null);
  const [errorBanner, setErrorBanner] = useState(null);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [memoryData, setMemoryData] = useState(null);
  const [leftPanel, setLeftPanel] = useState('trace'); // 'trace' or 'history'
  
  const logsEndRef = useRef(null);

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  useEffect(() => {
    loadHistory();
    loadMemory();
  }, []);

  const loadMemory = async () => {
    try {
      const response = await fetch('/api/memory');
      const data = await response.json();
      setMemoryData(data);
    } catch (e) {
      console.error("Failed to load memory", e);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await fetch('/api/history');
      const data = await response.json();
      setHistory(data || []);
      if (data && data.length > 0) {
        const savedId = localStorage.getItem('activeTripId');
        const active = data.find(t => t.id === savedId) || data[data.length - 1];
        displayTrip(active);
      }
    } catch (e) {
      console.error("Failed to load history", e);
    }
  };

  const displayTrip = (trip) => {
    if (!trip) return;
    localStorage.setItem('activeTripId', trip.id);
    setActiveTrip(trip);
    
    if (trip.events && trip.events.length > 0) {
      const formattedEvents = trip.events.map(ev => ({
        date: ev.date,
        title: ev.summary || (ev.info ? ev.info.split('\n')[0].replace(/\*\*/g, '').trim() : 'Event'),
        description: ev.info
      }));
      setCalendarEvents(formattedEvents);
      if (formattedEvents.length > 0) setCurrentDate(parseISO(formattedEvents[0].date));
    } else {
      setCalendarEvents([]); 
      setCurrentDate(new Date());
    }
  };

  const selectHistory = (id) => {
    const trip = history.find(t => t.id === id);
    if (trip) displayTrip(trip);
  };

  const deleteTrip = async (e, id) => {
    e.stopPropagation();
    try {
      await fetch(`/api/trip/${id}`, { method: 'DELETE' });
      setHistory(prev => prev.filter(t => t.id !== id));
      if (activeTrip?.id === id) {
        setActiveTrip(null);
        setCalendarEvents([]);
      }
    } catch (err) {
      console.error("Failed to delete trip", err);
    }
  };

  const startAgent = async () => {
    if (!input.trim() || isProcessing) return;
    const goal = input;
    setInput('');
    setLogs([]);
    setIsProcessing(true);
    setActiveTrip(null);
    setErrorBanner(null);
    setLeftPanel('trace');

    setLogs([{ id: 'goal', type: 'goal', content: goal, node: 'USER', timestamp: new Date() }]);

    let lastCount = 0;
    const poll = setInterval(async () => {
      try {
        const response = await fetch('/api/logs');
        const data = await response.json();
        const serverLogs = data.logs || [];
        if (serverLogs.length > lastCount) {
          lastCount = serverLogs.length;
          
          const formattedLogs = serverLogs.map((log, i) => ({
            id: `log-${i}`,
            type: determineLogType(log.node),
            content: log.content || log.event || "",
            node: log.node,
            duration: log.duration_ms,
            timestamp: new Date()
          }));
          
          setLogs(prev => {
            const newLogs = [...prev];
            return [newLogs[0], ...formattedLogs];
          });
        }
      } catch (error) {
        console.error("Polling error", error);
      }
    }, 800);

    try {
      const response = await fetch('/api/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: goal })
      });
      const data = await response.json();
      clearInterval(poll);

      if (data.error || !data.success) {
        setLogs(prev => [...prev, { id: 'err', type: 'error', content: data.error || 'Execution failed.', node: 'SYSTEM', timestamp: new Date() }]);
        setErrorBanner(data.error || 'Execution failed.');
        setIsProcessing(false);
        return;
      }

      if (data.is_infeasible) {
        setErrorBanner('Trip rejected: over budget or infeasible.');
      } else {
        if (data.trip && data.trip.events) {
          const formattedEvents = data.trip.events.map(ev => ({
            date: ev.date,
            title: ev.info ? ev.info.split('\n')[0] : 'Event',
            description: ev.info
          }));
          setCalendarEvents(formattedEvents);
          if (formattedEvents.length > 0) setCurrentDate(parseISO(formattedEvents[0].date));
        } else {
          setCalendarEvents([]);
          setCurrentDate(new Date());
        }

        setLogs(prev => [...prev, { id: 'fin', type: 'final_answer', content: 'Trip planning completed.', node: 'SYSTEM', timestamp: new Date() }]);
      }

      displayTrip(data.trip);
      loadHistory();
      loadMemory();

    } catch (error) {
      clearInterval(poll);
      setErrorBanner('Network error scheduling trip.');
      setLogs(prev => [...prev, { id: 'err', type: 'error', content: 'Network error.', node: 'SYSTEM', timestamp: new Date() }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const determineLogType = (node) => {
    if (node === 'PLANNER') return 'reasoning';
    if (node === 'PLAN_TRIP') return 'tool_call';
    if (node === 'ASSEMBLE') return 'final_answer';
    return 'observation';
  };

  const getIcon = (type) => {
    switch (type) {
      case 'goal': return <Play className="w-5 h-5 text-blue-500" />;
      case 'reasoning': return <FileText className="w-5 h-5 text-purple-500" />;
      case 'tool_call': return <FileText className="w-5 h-5 text-orange-500" />;
      case 'observation': return <ArrowRight className="w-5 h-5 text-green-500" />;
      case 'final_answer': return <CheckCircle2 className="w-5 h-5 text-emerald-500" />;
      case 'error': return <AlertCircle className="w-5 h-5 text-red-500" />;
      default: return <FileText className="w-5 h-5" />;
    }
  };

  // Calendar helpers
  const prevMonth = () => setCurrentDate(subMonths(currentDate, 1));
  const nextMonth = () => setCurrentDate(addMonths(currentDate, 1));
  const goToday = () => setCurrentDate(new Date());

  const monthStart = startOfMonth(currentDate);
  const monthEnd = endOfMonth(monthStart);
  const calStart = startOfWeek(monthStart);
  const calEnd = endOfWeek(monthEnd);
  const calDays = eachDayOfInterval({ start: calStart, end: calEnd });
  const weekDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans">
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm relative z-20">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <MapPin className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-900">Travel Agent AI</h1>
            <p className="text-sm text-gray-500">Intelligent Travel Concierge</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <div className="flex space-x-2">
            <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 text-xs font-semibold transition-colors" onClick={() => { setInput('Plan a 5-day trip to Paris from Delhi for ₹30000'); }}>Test Rejection</button>
            <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 text-xs font-semibold transition-colors" onClick={() => { setInput('Plan a 3-day Rajasthan trip from Delhi for ₹60000'); }}>Test Approval</button>
          </div>
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            System Online
          </span>
        </div>
      </header>

      <main className="flex-1 overflow-hidden flex flex-col md:flex-row relative z-10">
        
        {/* Left Panel: Trace / History / Memory */}
        <div className="w-full md:w-80 lg:w-96 flex flex-col border-r border-gray-200 bg-white flex-shrink-0">
          {/* Left panel tabs */}
          <div className="flex border-b border-gray-200 bg-gray-50">
            <button
              onClick={() => setLeftPanel('trace')}
              className={`flex-1 py-3 px-3 text-xs font-semibold uppercase tracking-wider transition-colors border-b-2 ${leftPanel === 'trace' ? 'border-blue-600 text-blue-600 bg-white' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              <span className="flex items-center justify-center gap-1.5">
                <FileText className="w-3.5 h-3.5" /> Trace
              </span>
            </button>
            <button
              onClick={() => setLeftPanel('history')}
              className={`flex-1 py-3 px-3 text-xs font-semibold uppercase tracking-wider transition-colors border-b-2 ${leftPanel === 'history' ? 'border-blue-600 text-blue-600 bg-white' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              <span className="flex items-center justify-center gap-1.5">
                <Clock className="w-3.5 h-3.5" /> History
                {history.length > 0 && <span className="bg-blue-100 text-blue-700 text-[10px] font-bold px-1.5 py-0.5 rounded-full ml-1">{history.length}</span>}
              </span>
            </button>
            <button
              onClick={() => setLeftPanel('memory')}
              className={`flex-1 py-3 px-3 text-xs font-semibold uppercase tracking-wider transition-colors border-b-2 ${leftPanel === 'memory' ? 'border-blue-600 text-blue-600 bg-white' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              <span className="flex items-center justify-center gap-1.5">
                <Brain className="w-3.5 h-3.5" /> Memory
              </span>
            </button>
          </div>
          
          {/* LEFT PANEL CONTENT */}
          <div className="flex-1 overflow-y-auto">

            {/* TRACE TAB */}
            {leftPanel === 'trace' && (
              <div className="p-4 space-y-4">
                {logs.length === 0 ? (
                  <div className="text-center text-sm text-gray-400 mt-10">No active processes.</div>
                ) : null}
                {logs.map((log) => (
                  <div key={log.id} className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    <div className="mt-0.5 flex-shrink-0">
                      {getIcon(log.type)}
                    </div>
                    <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3 shadow-sm text-sm">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-blue-600">
                          {log.node} {log.duration ? `(${log.duration}ms)` : ''}
                        </span>
                        <span className="text-[10px] text-gray-400">
                          {log.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="text-xs text-gray-800 font-mono whitespace-pre-wrap break-words">
                        {log.type === 'reasoning' || log.type === 'final_answer' || log.type === 'goal' ? (
                          log.content
                        ) : (
                          <div className="opacity-70">{log.content.substring(0, 150)}{log.content.length > 150 ? '...' : ''}</div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}

            {/* HISTORY TAB */}
            {leftPanel === 'history' && (
              <div className="p-3 space-y-2">
                {history.length === 0 ? (
                  <div className="text-center text-sm text-gray-400 mt-10">No trip history yet.</div>
                ) : (
                  [...history].reverse().map((trip) => {
                    const isActive = activeTrip?.id === trip.id;
                    const dest = trip.destination || trip.goal?.destination || 'Unknown';
                    const date = trip.created_at ? format(parseISO(trip.created_at), 'MMM d, yyyy') : trip.date || '';
                    const budget = trip.goal?.budget_inr;
                    const days = trip.goal?.duration_days;
                    return (
                      <div
                        key={trip.id}
                        onClick={() => selectHistory(trip.id)}
                        className={`group p-3 rounded-lg border cursor-pointer transition-all duration-200 ${isActive ? 'border-blue-400 bg-blue-50 shadow-sm' : 'border-gray-200 bg-white hover:border-blue-200 hover:bg-blue-50/30'}`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <MapPin className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-blue-600' : 'text-gray-400'}`} />
                              <span className={`text-sm font-semibold truncate ${isActive ? 'text-blue-700' : 'text-gray-800'}`}>{dest}</span>
                            </div>
                            <div className="flex items-center gap-3 mt-1.5 ml-5">
                              {days && <span className="text-[10px] text-gray-500">{days} days</span>}
                              {budget && <span className="text-[10px] text-gray-500">₹{Number(budget).toLocaleString()}</span>}
                              {date && <span className="text-[10px] text-gray-400">{date}</span>}
                            </div>
                          </div>
                          <button
                            onClick={(e) => deleteTrip(e, trip.id)}
                            className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-all"
                            title="Delete trip"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            )}

            {/* MEMORY TAB */}
            {leftPanel === 'memory' && (
              <div className="p-4 space-y-4">
                {memoryData ? (
                  <>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-sm">
                        <User className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-gray-900">Traveler Profile</p>
                        <p className="text-[11px] text-gray-400">Learned from your trips</p>
                      </div>
                    </div>
                    {[
                      { label: 'Home City', value: memoryData.home_city, emoji: '🏠' },
                      { label: 'Nationality', value: memoryData.nationality, emoji: '🌍' },
                      { label: 'Diet', value: memoryData.diet, emoji: '🍽️' },
                      { label: 'Budget Style', value: memoryData.budget_style, emoji: '💰' },
                      { label: 'Last Destination', value: memoryData.last_destination, emoji: '📍' },
                    ].map(({ label, value, emoji }) => (
                      <div key={label} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-100">
                        <span className="text-xs text-gray-500 flex items-center gap-2"><span>{emoji}</span>{label}</span>
                        <span className="text-xs font-semibold text-gray-800">{value || '—'}</span>
                      </div>
                    ))}
                    {memoryData.past_trips && memoryData.past_trips.length > 0 && (
                      <div className="mt-4">
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Past Trips ({memoryData.past_trips.length})</p>
                        <div className="space-y-1.5">
                          {memoryData.past_trips.slice(-5).reverse().map((pt, i) => (
                            <div key={i} className="flex items-center justify-between p-2 bg-white rounded border border-gray-100 text-xs">
                              <span className="font-medium text-gray-700">{pt.destination}</span>
                              <div className="flex items-center gap-2 text-gray-400">
                                <span>₹{Number(pt.budget_inr).toLocaleString()}</span>
                                <span>{pt.date}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center text-sm text-gray-400 mt-10">Loading memory...</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Main Area: Itinerary/Calendar & Input */}
        <div className="flex-1 flex flex-col bg-white relative">
          
          {errorBanner && (
             <div className="bg-red-50 border-l-4 border-red-500 p-4 absolute top-4 left-6 right-6 z-30 flex items-center shadow-md rounded-r-lg animate-in fade-in slide-in-from-top-2">
                <AlertCircle className="text-red-500 w-5 h-5 mr-3" />
                <p className="text-red-700 text-sm font-medium">{errorBanner}</p>
                <button className="ml-auto text-red-500" onClick={() => setErrorBanner(null)}>×</button>
             </div>
          )}

          <div className="flex-1 overflow-y-auto flex flex-col">
            <div className="sticky top-0 bg-white/90 backdrop-blur-md z-10 border-b border-gray-200 px-6 pt-4 flex justify-between items-center">
              <div className="flex gap-4">
                  <button 
                    onClick={() => setActiveTab('itinerary')} 
                    className={`pb-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'itinerary' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
                  >
                    Itinerary Map
                  </button>
                  <button 
                    onClick={() => setActiveTab('calendar')} 
                    className={`pb-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'calendar' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
                  >
                    Visual Calendar
                  </button>
              </div>
              {activeTrip && activeTrip.ics_path && (
                  <a href={`/itineraries/${activeTrip.ics_path}`} className="pb-2 text-sm font-medium text-blue-600 hover:text-blue-800 flex items-center gap-1 transition-colors" download>
                      <Download className="w-4 h-4" /> Export .ICS
                  </a>
              )}
            </div>

            <div className="flex-1 p-6 max-w-4xl mx-auto w-full pb-40">
              {activeTab === 'itinerary' ? (
                <>
                  {activeTrip ? (
                    <div className="prose prose-blue max-w-none text-gray-800">
                      <Markdown remarkPlugins={[remarkGfm]}>{activeTrip.itinerary}</Markdown>
                    </div>
                  ) : isProcessing ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400 mt-20 animate-pulse">
                      <Loader2 className="w-12 h-12 mb-4 animate-spin opacity-50 text-blue-500" />
                      <p className="text-lg font-medium text-gray-600">Agents are coordinating your trip...</p>
                      <p className="text-sm mt-2 opacity-60">Checking flights, hotels, and local activities</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400 mt-20">
                      <MapPin className="w-16 h-16 mb-4 opacity-20" />
                      <p className="text-lg">What kind of trip are you planning?</p>
                    </div>
                  )}
                </>
              ) : (
                /* ═══════════ FULL CALENDAR VIEW ═══════════ */
                <div className="flex-1 flex flex-col h-full bg-white rounded-xl border border-gray-200 overflow-hidden relative">
                  {/* Calendar Header */}
                  <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-gray-50/50">
                    <h3 className="text-2xl font-semibold text-gray-900">{format(monthStart, 'MMMM yyyy')}</h3>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={goToday}
                        className="px-3 py-1.5 text-xs font-semibold border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600 transition-colors shadow-sm text-gray-600"
                      >
                        Today
                      </button>
                      <button onClick={prevMonth} className="p-2 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors shadow-sm">
                        <ChevronLeft className="w-5 h-5 text-gray-600" />
                      </button>
                      <button onClick={nextMonth} className="p-2 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors shadow-sm">
                        <ChevronRight className="w-5 h-5 text-gray-600" />
                      </button>
                    </div>
                  </div>

                  {/* Weekday headers */}
                  <div className="grid grid-cols-7 border-b border-gray-100">
                    {weekDays.map(day => (
                      <div key={day} className="text-center text-xs font-semibold text-gray-500 py-2.5 uppercase tracking-wider">{day}</div>
                    ))}
                  </div>

                  {/* Calendar grid */}
                  <div className="grid grid-cols-7 flex-1 auto-rows-[minmax(90px,1fr)]">
                    {calDays.map(day => {
                      const dayString = format(day, 'yyyy-MM-dd');
                      const dayEvents = calendarEvents.filter(e => e.date === dayString);
                      const inMonth = isSameMonth(day, monthStart);
                      const today = isToday(day);
                      return (
                        <div
                          key={day.toString()}
                          className={`border-b border-r border-gray-100 p-1.5 flex flex-col transition-colors
                            ${!inMonth ? 'bg-gray-50/70' : 'bg-white'}
                            ${dayEvents.length > 0 ? 'bg-blue-50/60' : ''}
                            ${today ? 'ring-2 ring-inset ring-blue-400' : ''}
                          `}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className={`text-xs font-medium w-6 h-6 flex items-center justify-center rounded-full
                              ${today ? 'bg-blue-600 text-white' : ''}
                              ${!today && dayEvents.length > 0 ? 'text-blue-700 font-bold' : ''}
                              ${!today && dayEvents.length === 0 && inMonth ? 'text-gray-600' : ''}
                              ${!inMonth && !today ? 'text-gray-300' : ''}
                            `}>
                              {format(day, 'd')}
                            </span>
                          </div>
                          <div className="flex-1 overflow-y-auto space-y-1">
                            {dayEvents.map((evt, i) => (
                              <div
                                key={i}
                                className="text-[10px] leading-tight bg-blue-100 text-blue-800 px-1.5 py-1 rounded border border-blue-200 cursor-default hover:bg-blue-200 transition-colors"
                                title={evt.description}
                              >
                                <div className="font-semibold truncate">{evt.title}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Bottom Input Area */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white/95 to-transparent pt-12 pb-6 px-6 pointer-events-none z-20">
            <div className="max-w-4xl mx-auto relative pointer-events-auto">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    startAgent();
                  }
                }}
                placeholder="e.g. Plan a 5-day trip to Paris from Delhi for ₹40000..."
                className="w-full pl-6 pr-16 py-4 bg-white border border-gray-300 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.08)] focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none h-16 text-base outline-none transition-shadow"
                disabled={isProcessing}
              />
              <button
                onClick={startAgent}
                disabled={isProcessing || !input.trim()}
                className="absolute right-2 top-2 bottom-2 aspect-square flex items-center justify-center bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg"
              >
                {isProcessing ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5 ml-1" />
                )}
              </button>
            </div>
            <div className="text-center mt-3 text-[11px] font-medium text-gray-400 pointer-events-auto uppercase tracking-wider">
              Powered by Multi-Agent Architecture
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
