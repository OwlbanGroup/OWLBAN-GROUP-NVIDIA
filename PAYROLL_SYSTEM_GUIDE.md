# Complete Payroll System Guide

## Overview

This guide covers the complete payroll system implementation including backend API, frontend UI, and integration with JPMorgan ACH payments.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PAYROLL SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frontend (Next.js)          Backend (NestJS)               │
│  ┌──────────────────┐       ┌──────────────────┐          │
│  │ Payroll UI       │──────→│ PayrollModule    │          │
│  │ - Dashboard      │       │ - Controller     │          │
│  │ - Employees      │       │ - Service        │          │
│  │ - Runs           │       │ - Entities       │          │
│  │ - Execute        │       └────────┬─────────┘          │
│  └──────────────────┘                │                     │
│                                      │                     │
│                            ┌─────────▼─────────┐          │
│                            │ PaymentsService   │          │
│                            │ - sendAchPayment  │          │
│                            │ - getStatus       │          │
│                            └─────────┬─────────┘          │
│                                      │                     │
│                            ┌─────────▼─────────┐          │
│                            │ JPMorgan API      │          │
│                            │ - ACH Payments    │          │
│                            └───────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Database Schema

### Employee Table
```sql
CREATE TABLE employees (
  id UUID PRIMARY KEY,
  organization_id UUID REFERENCES organizations(id),
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL,
  bank_routing_number VARCHAR(9) NOT NULL,
  bank_account_number VARCHAR(255) NOT NULL,
  pay_rate NUMERIC(12,2) NOT NULL,
  pay_frequency VARCHAR(20) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### PayrollRun Table
```sql
CREATE TABLE payroll_runs (
  id UUID PRIMARY KEY,
  organization_id UUID REFERENCES organizations(id),
  run_date TIMESTAMPTZ NOT NULL,
  period_start DATE NOT NULL,
  period_end DATE NOT NULL,
  status VARCHAR(20) DEFAULT 'PENDING',
  total_gross NUMERIC(14,2),
  total_net NUMERIC(14,2),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### PayrollPayment Table
```sql
CREATE TABLE payroll_payments (
  id UUID PRIMARY KEY,
  payroll_run_id UUID REFERENCES payroll_runs(id),
  employee_id UUID REFERENCES employees(id),
  gross_pay NUMERIC(12,2) NOT NULL,
  net_pay NUMERIC(12,2) NOT NULL,
  jpm_payment_id VARCHAR(255),
  status VARCHAR(20) DEFAULT 'PENDING',
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Backend API Endpoints

### Employee Management

#### Add Employee
```http
POST /api/payroll/employee/:orgId
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "bankRoutingNumber": "123456789",
  "bankAccountNumber": "9876543210",
  "payRate": 5000.00,
  "payFrequency": "BIWEEKLY"
}
```

#### List Employees
```http
GET /api/payroll/employees/:orgId
```

### Payroll Run Management

#### Create Payroll Run
```http
POST /api/payroll/run/:orgId
Content-Type: application/json

{
  "periodStart": "2024-01-01",
  "periodEnd": "2024-01-15"
}
```

**Process:**
1. Fetches all employees for the organization
2. Creates a PayrollRun record with status PENDING
3. For each employee:
   - Calculates gross pay from payRate
   - Calculates net pay (gross * 0.92 - 8% withholding)
   - Creates PayrollPayment record
4. Updates run with totalGross and totalNet
5. Returns the created run

#### List Payroll Runs
```http
GET /api/payroll/runs/:orgId
```

#### Get Single Run with Payments
```http
GET /api/payroll/run/:runId
```

Returns run with all payments and employee details.

### Payroll Execution

#### Execute Payroll Run
```http
POST /api/payroll/execute/:runId
Content-Type: application/json

{
  "debitAccountId": "account-uuid-here"
}
```

**Process:**
1. Loads payroll run with all payments and employees
2. Sets run status to PROCESSING
3. For each payment:
   - Calls PaymentsService.sendAchPayment()
   - Initiates ACH transfer via JPMorgan API
   - Stores JPMorgan payment ID
   - Updates payment status to SENT
4. Sets run status to COMPLETED
5. Returns updated run

## Frontend UI Structure

### File Structure
```
frontend-example/
├── lib/
│   └── api.ts                    # API helper functions
├── components/
│   └── Payroll/
│       ├── EmployeeForm.tsx      # Add employee form
│       ├── EmployeeTable.tsx     # Employee list table
│       ├── PayrollRunForm.tsx    # Create run form
│       ├── PayrollRunTable.tsx   # Runs list table
│       └── PayrollRunDetail.tsx  # Run detail & execute
└── app/
    └── payroll/
        ├── layout.tsx            # Payroll layout with nav
        ├── page.tsx              # Dashboard
        ├── employees/
        │   └── page.tsx          # Employee management
        ├── runs/
        │   ├── page.tsx          # Runs list
        │   └── [runId]/
        │       └── page.tsx      # Run detail
```

### Pages

#### 1. Payroll Dashboard (`/payroll`)
- Shows employee count
- Displays last payroll run
- Lists recent runs (last 5)

#### 2. Employee Management (`/payroll/employees`)
- Form to add new employees
- Table showing all employees with:
  - Name, email
  - Pay rate, frequency
  - Date added

#### 3. Payroll Runs (`/payroll/runs`)
- Form to create new payroll run
- Table showing all runs with:
  - Run date, period
  - Total gross/net
  - Status
  - Link to view/execute

#### 4. Run Detail (`/payroll/runs/[runId]`)
- Run information (date, period, status, totals)
- Table of all payments with:
  - Employee name, email
  - Gross pay, net pay
  - Payment status
  - JPMorgan payment ID
- Execute button (if status is PENDING)
  - Input for debit account ID
  - Initiates ACH payments

## Usage Flow

### 1. Add Employees
```bash
# Navigate to /payroll/employees
# Fill in employee form:
- Name: John Doe
- Email: john@example.com
- Routing #: 123456789
- Account #: 9876543210
- Pay rate: 5000.00
- Frequency: BIWEEKLY
# Click "Add employee"
```

### 2. Create Payroll Run
```bash
# Navigate to /payroll/runs
# Fill in run form:
- Period start: 2024-01-01
- Period end: 2024-01-15
# Click "Create run"
# System automatically:
- Calculates gross/net for all employees
- Creates payment records
- Sets status to PENDING
```

### 3. Review Run
```bash
# Click "View / execute" on the run
# Review:
- Run details (date, period, totals)
- All employee payments
- Gross and net amounts
```

### 4. Execute Payroll
```bash
# On run detail page:
# Enter debit account ID
# Click "Execute payroll"
# System:
- Initiates ACH payment for each employee
- Stores JPMorgan payment IDs
- Updates payment status to SENT
- Sets run status to COMPLETED
```

## Configuration

### Backend (.env)
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=owldashboard

# API
PORT=4000
API_PREFIX=api
API_VERSION=v1

# JPMorgan
JPM_API_BASE_URL=https://sandbox.jpmorgan.com/api
JPM_API_KEY=your-api-key
```

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_BASE=http://localhost:4000/api
```

## Integration with JPMorgan

### PaymentsService

The `PaymentsService` handles ACH payment initiation:

```typescript
async sendAchPayment(dto: {
  fromAccountId: string;
  toRouting: string;
  toAccount: string;
  amount: string;
  memo: string;
}) {
  // TODO: Replace with actual JPMorgan Payments API call
  // Current implementation is a placeholder
  
  const paymentId = 'JPM-' + Math.random().toString(36).substring(2, 15);
  
  return {
    id: paymentId,
    status: 'SENT',
    amount: dto.amount,
    fromAccountId: dto.fromAccountId,
    toRouting: dto.toRouting,
    toAccount: dto.toAccount,
    memo: dto.memo,
    createdAt: new Date().toISOString(),
  };
}
```

### Real JPMorgan Integration

To integrate with real JPMorgan Payments API:

1. **Update PaymentsService:**
```typescript
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';

async sendAchPayment(dto: any) {
  const apiUrl = this.config.get('JPM_API_BASE_URL');
  const apiKey = this.config.get('JPM_API_KEY');
  
  const response = await this.http.post(
    `${apiUrl}/payments/ach`,
    {
      debitAccount: dto.fromAccountId,
      creditAccount: dto.toAccount,
      routingNumber: dto.toRouting,
      amount: dto.amount,
      memo: dto.memo,
    },
    {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
    },
  ).toPromise();
  
  return response.data;
}
```

2. **Add error handling:**
```typescript
try {
  const jpmPayment = await this.paymentsService.sendAchPayment({...});
  payment.jpmPaymentId = jpmPayment.id;
  payment.status = 'SENT';
} catch (error) {
  payment.status = 'FAILED';
  this.logger.error(`Payment failed for employee ${emp.id}:`, error);
}
```

## Security Considerations

1. **Authentication:** Add JWT authentication to all endpoints
2. **Authorization:** Verify user has access to organization
3. **Encryption:** Encrypt bank account numbers at rest
4. **Audit Logging:** Log all payroll operations
5. **Rate Limiting:** Prevent abuse of payment endpoints
6. **Validation:** Validate all input data (routing numbers, amounts)

## Testing

### Backend Tests
```bash
cd nestjs-backend
npm run test
```

### Frontend Tests
```bash
cd frontend-example
npm run test
```

### Manual Testing

1. **Add Employee:**
```bash
curl -X POST http://localhost:4000/api/payroll/employee/ORG_ID \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Employee",
    "email": "test@example.com",
    "bankRoutingNumber": "123456789",
    "bankAccountNumber": "9876543210",
    "payRate": 5000,
    "payFrequency": "BIWEEKLY"
  }'
```

2. **Create Run:**
```bash
curl -X POST http://localhost:4000/api/payroll/run/ORG_ID \
  -H "Content-Type: application/json" \
  -d '{
    "periodStart": "2024-01-01",
    "periodEnd": "2024-01-15"
  }'
```

3. **Execute Run:**
```bash
curl -X POST http://localhost:4000/api/payroll/execute/RUN_ID \
  -H "Content-Type: application/json" \
  -d '{
    "debitAccountId": "ACCOUNT_ID"
  }'
```

## Deployment

### Backend
```bash
cd nestjs-backend
npm run build
npm run start:prod
```

### Frontend
```bash
cd frontend-example
npm run build
npm run start
```

### Docker
```bash
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **"ORG_UUID_HERE" error:**
   - Replace hardcoded ORG_ID with real organization ID
   - Implement proper authentication to get org from JWT

2. **Payment execution fails:**
   - Check JPMorgan API credentials
   - Verify bank account details are valid
   - Check PaymentsService logs

3. **Frontend can't connect to backend:**
   - Verify NEXT_PUBLIC_API_BASE in .env.local
   - Check backend is running on port 4000
   - Verify CORS is configured correctly

## Next Steps

1. **Replace hardcoded ORG_ID** with real org from JWT authentication
2. **Add account selector** instead of free-text debit account ID
3. **Implement real JPMorgan API** integration
4. **Add styling** with Tailwind CSS or Chakra UI
5. **Add validation** for routing numbers and account numbers
6. **Implement webhooks** to receive payment status updates
7. **Add reporting** for payroll history and analytics
8. **Implement tax calculations** for different jurisdictions
9. **Add employee self-service** portal
10. **Implement approval workflow** for payroll runs

## Support

For issues or questions:
- Backend: Check NestJS logs
- Frontend: Check browser console
- Database: Check PostgreSQL logs
- Payments: Check JPMorgan API documentation
