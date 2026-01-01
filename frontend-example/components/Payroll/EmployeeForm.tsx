'use client';

import { useState } from 'react';
import { apiPost } from '../../lib/api';

type Props = {
  orgId: string;
  onCreated?: () => void;
};

export function EmployeeForm({ orgId, onCreated }: Props) {
  const [form, setForm] = useState({
    name: '',
    email: '',
    bankRoutingNumber: '',
    bankAccountNumber: '',
    payRate: '',
    payFrequency: 'BIWEEKLY',
  });
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      await apiPost(`/payroll/employee/${orgId}`, {
        ...form,
        payRate: Number(form.payRate),
      });
      setForm({
        name: '',
        email: '',
        bankRoutingNumber: '',
        bankAccountNumber: '',
        payRate: '',
        payFrequency: 'BIWEEKLY',
      });
      onCreated?.();
    } catch (err) {
      console.error(err);
      alert('Failed to create employee');
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ marginBottom: 24 }}>
      <h3>Add Employee</h3>
      <div>
        <label>
          Name
          <input
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            required
          />
        </label>
      </div>
      <div>
        <label>
          Email
          <input
            type="email"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            required
          />
        </label>
      </div>
      <div>
        <label>
          Routing #
          <input
            value={form.bankRoutingNumber}
            onChange={(e) =>
              setForm({ ...form, bankRoutingNumber: e.target.value })
            }
            required
          />
        </label>
      </div>
      <div>
        <label>
          Account #
          <input
            value={form.bankAccountNumber}
            onChange={(e) =>
              setForm({ ...form, bankAccountNumber: e.target.value })
            }
            required
          />
        </label>
      </div>
      <div>
        <label>
          Pay rate (per period)
          <input
            type="number"
            step="0.01"
            value={form.payRate}
            onChange={(e) => setForm({ ...form, payRate: e.target.value })}
            required
          />
        </label>
      </div>
      <div>
        <label>
          Frequency
          <select
            value={form.payFrequency}
            onChange={(e) => setForm({ ...form, payFrequency: e.target.value })}
          >
            <option value="WEEKLY">Weekly</option>
            <option value="BIWEEKLY">Biweekly</option>
            <option value="MONTHLY">Monthly</option>
          </select>
        </label>
      </div>
      <button type="submit" disabled={loading}>
        {loading ? 'Saving...' : 'Add employee'}
      </button>
    </form>
  );
}
