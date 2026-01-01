import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { JpmorganService } from './jpmorgan.service';

@Module({
  imports: [
    HttpModule.register({
      timeout: 5000,
    }),
  ],
  providers: [JpmorganService],
  exports: [JpmorganService],
})
export class JpmorganModule {}
