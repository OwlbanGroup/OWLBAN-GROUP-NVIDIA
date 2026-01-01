import { apiGet } from '../../lib/api';
import { PayrollRunTable } from '../../components/Payroll/PayrollRunTable';

const ORG_ID = 'ORG_UUID_HERE'; // replace with real org from auth

async function getData() {
  const runs = await apiGet<any[]>(`/payroll/runs/${ORG_ID}`);
  const employees = await apiGet<any[]>(`/payroll/employees/${ORG_ID}`);
  return { runs, employees };
}

export default async function PayrollDashboardPage() {
  const { runs, employees } = await getData();

  const totalEmployees = employees.length;
  const lastRun = runs[0];

  return (
    <div>
      <h2>Overview</h2>
      <p>
        Employees: {totalEmployees} <br />
        Last run:{' '}
        {lastRun
          ? `${new Date(lastRun.runDate).toLocaleString()} (${lastRun.status})`
          : 'No runs yet'}
      </p>

      <h3>Recent runs</h3>
      <PayrollRunTable runs={runs.slice(0, 5)} />
    </div>
  );
}
